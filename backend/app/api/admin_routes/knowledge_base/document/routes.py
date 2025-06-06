import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi_pagination import Params, Page
from sqlmodel import Session

from app.api.admin_routes.knowledge_base.models import ChunkItem
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import Document
from app.models.chunk import KgIndexStatus, get_kb_chunk_model
from app.models.document import DocIndexTaskStatus
from app.models.entity import get_kb_entity_model
from app.models.relationship import get_kb_relationship_model
from app.repositories import knowledge_base_repo, document_repo
from app.repositories.chunk import ChunkRepo
from app.api.admin_routes.knowledge_base.document.models import (
    DocumentFilters,
    DocumentItem,
    RebuildIndexResult,
)
from app.exceptions import InternalServerError
from app.repositories.graph import GraphRepo
from app.tasks.build_index import build_index_for_document, build_kg_index_for_chunk
from app.tasks.knowledge_base import stats_for_knowledge_base


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/knowledge_bases/{kb_id}/documents")
def list_kb_documents(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    filters: Annotated[DocumentFilters, Query()],
    params: Params = Depends(),
) -> Page[DocumentItem]:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        filters.knowledge_base_id = kb.id
        return document_repo.paginate(
            session=session,
            filters=filters,
            params=params,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/admin/knowledge_bases/{kb_id}/documents/{doc_id}")
def get_kb_document_by_id(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    doc_id: int,
) -> Document:
    try:
        document = document_repo.must_get(session, doc_id)
        assert document.knowledge_base_id == kb_id
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/admin/knowledge_bases/{kb_id}/documents/{doc_id}/chunks")
def list_kb_document_chunks(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    doc_id: int,
) -> list[ChunkItem]:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        chunk_repo = ChunkRepo(get_kb_chunk_model(kb))
        return chunk_repo.get_document_chunks(session, doc_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.delete("/admin/knowledge_bases/{kb_id}/documents/{document_id}")
def remove_kb_document(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    document_id: int,
) -> RebuildIndexResult:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        doc = document_repo.must_get(session, document_id)
        assert doc.knowledge_base_id == kb.id

        chunk_model = get_kb_chunk_model(kb)
        entity_model = get_kb_entity_model(kb)
        relationship_model = get_kb_relationship_model(kb)

        chunk_repo = ChunkRepo(chunk_model)
        graph_repo = GraphRepo(entity_model, relationship_model, chunk_model)

        graph_repo.delete_document_relationships(session, document_id)
        logger.info(
            f"Deleted relationships generated by document #{document_id} successfully."
        )

        graph_repo.delete_orphaned_entities(session)
        logger.info("Deleted orphaned entities successfully.")

        chunk_repo.delete_by_document(session, document_id)
        logger.info(f"Deleted chunks of document #{document_id} successfully.")

        session.delete(doc)
        session.commit()

        stats_for_knowledge_base.delay(kb_id)

        return {"detail": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to remove document #{document_id}: {e}")
        raise InternalServerError()


@router.post("/admin/knowledge_bases/{kb_id}/documents/reindex")
def rebuild_kb_documents_index(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    document_ids: list[int],
    reindex_completed_task: bool = False,
):
    try:
        return rebuild_kb_document_index_by_ids(
            session, kb_id, document_ids, reindex_completed_task
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e, exc_info=True)
        raise InternalServerError()


@router.post("/admin/knowledge_bases/{kb_id}/documents/{doc_id}/reindex")
def rebuild_kb_document_index(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    doc_id: int,
    reindex_completed_task: bool = False,
) -> RebuildIndexResult:
    try:
        document_ids = [doc_id]
        return rebuild_kb_document_index_by_ids(
            db_session, kb_id, document_ids, reindex_completed_task
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e, exc_info=True)
        raise InternalServerError()


def rebuild_kb_document_index_by_ids(
    db_session: Session,
    kb_id: int,
    document_ids: list[int],
    reindex_completed_task: bool = False,
) -> RebuildIndexResult:
    kb = knowledge_base_repo.must_get(db_session, kb_id)
    kb_chunk_repo = ChunkRepo(get_kb_chunk_model(kb))

    # Retry failed vector index tasks.
    documents = document_repo.fetch_by_ids(db_session, document_ids)
    reindex_document_ids = []
    ignore_document_ids = []

    for doc in documents:
        # TODO: check NOT_STARTED, PENDING, RUNNING
        if doc.index_status != DocIndexTaskStatus.FAILED and not reindex_completed_task:
            ignore_document_ids.append(doc.id)
        else:
            reindex_document_ids.append(doc.id)

        doc.index_status = DocIndexTaskStatus.PENDING
        db_session.add(doc)
        db_session.commit()

        build_index_for_document.delay(kb.id, doc.id)

    # Retry failed kg index tasks.
    chunks = kb_chunk_repo.fetch_by_document_ids(db_session, document_ids)
    reindex_chunk_ids = []
    ignore_chunk_ids = []
    for chunk in chunks:
        if chunk.index_status == KgIndexStatus.COMPLETED and not reindex_completed_task:
            ignore_chunk_ids.append(chunk.id)
            continue
        else:
            reindex_chunk_ids.append(chunk.id)

        chunk.index_status = KgIndexStatus.PENDING
        db_session.add(chunk)
        db_session.commit()

        build_kg_index_for_chunk.delay(kb.id, chunk.id)

    return RebuildIndexResult(
        reindex_document_ids=reindex_document_ids,
        ignore_document_ids=ignore_document_ids,
        reindex_chunk_ids=reindex_chunk_ids,
        ignore_chunk_ids=ignore_chunk_ids,
    )
