from typing import Annotated

from fastapi import APIRouter, Depends

from api.auth import User, current_user

router = APIRouter()


@router.get("/v1/me")
def me(user: Annotated[User, Depends(current_user)]) -> User:
    """The signed-in caller and their Cognito groups (any group, or none)."""
    return user
