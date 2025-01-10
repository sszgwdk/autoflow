from fastapi import APIRouter, Depends
from fastapi_pagination import Params
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
) -> list[UserDescriptor]:
    query = select(User).order_by(User.id)
    if search:
        query = query.where(col(User.email).contains(search))
    users = session.exec(query).all()
    return [
        UserDescriptor(
            id=user.id,
            email=user.email
        )
        for user in users
    ]
