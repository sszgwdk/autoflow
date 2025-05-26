import logging

import pytest

from autoflow.configs.knowledge_base import IndexMethod
from autoflow.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def kb(db_engine, llm, embedding_model):
    kb = KnowledgeBase(
        db_engine=db_engine,
        name="Test_fts",
        description="Here is a knowledge base use full text search.",
        index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.FULLTEXT_SEARCH],
        llm=llm,
        embedding_model=embedding_model,
    )
    logger.info("Created a knowledge base successfully.")
    return kb


def test_add_documents_via_filepath(kb: KnowledgeBase):
    docs = kb.add("./tests/fixtures/analyze-slow-queries.md")
    assert len(docs) == 1


def test_search_documents(kb):
    result = kb.search_documents(
        query="What is TiDB?",
        mode="full_text",
        similarity_top_k=2,
    )
    assert len(result.chunks) > 0
