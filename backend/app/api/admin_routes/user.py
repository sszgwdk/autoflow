from fastapi import APIRouter, Depends
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import select, col

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import User

from app.api.admin_routes.models import (
    UserDescriptor,
)

router = APIRouter()


@router.get("/admin/users/search")
def search_users(
    session: SessionDep,
    user: CurrentSuperuserDep,
    search: str | None = None,
    params: Params = Depends(),
) -> Page[UserDescriptor]:
    query = select(User).order_by(User.id)
    if search:
        query = query.where(col(User.email).contains(search))
    return paginate(
        session,
        query,
        params,
        transformer=lambda items: [
            UserDescriptor(
                id=item.id,
                email=item.email,
            )
            for item in items
        ],
    )
