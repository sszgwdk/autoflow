import enum
from uuid import UUID
from typing import Optional
from pydantic import BaseModel
from datetime import datetime

from sqlmodel import (
    Field,
    Relationship as SQLRelationship,
)

from .base import UpdatableBaseModel


class FeedbackType(str, enum.Enum):
    LIKE = "like"
    DISLIKE = "dislike"

    @classmethod
    def adjust_relationship_weight(cls, feedback_type):
        weights = {cls.LIKE: 10, cls.DISLIKE: -10}
        return weights.get(feedback_type, 0)


class BaseFeedback(UpdatableBaseModel):
    feedback_type: FeedbackType = FeedbackType.LIKE
    comment: str = Field(max_length=500, default=None)
    chat_id: UUID
    chat_message_id: int
    user_id: UUID
    origin: Optional[str] = Field(max_length=256, default=None, nullable=True)


class Feedback(BaseFeedback, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_id: UUID = Field(foreign_key="chats.id")
    chat: "Chat" = SQLRelationship(  # noqa:F821
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Feedback.chat_id == Chat.id",
        },
    )
    chat_message_id: int = Field(foreign_key="chat_messages.id")
    chat_message: "ChatMessage" = SQLRelationship(  # noqa:F821
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Feedback.chat_message_id == ChatMessage.id",
        },
    )
    user_id: UUID = Field(foreign_key="users.id", nullable=True)
    user: "User" = SQLRelationship(  # noqa:F821
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Feedback.user_id == User.id",
        },
    )

    __tablename__ = "feedbacks"


class AdminFeedbackPublic(BaseFeedback):
    id: int
    chat_title: str
    chat_origin: Optional[str]
    chat_message_content: str
    user_id: Optional[UUID]
    user_email: Optional[str]


class FeedbackFilters(BaseModel):
    created_at_start: Optional[datetime] = None
    created_at_end: Optional[datetime] = None
    feedback_origin: Optional[str] = None
    chat_id: Optional[UUID] = None
    feedback_type: Optional[FeedbackType] = None
    user_id: Optional[UUID] = None


class FeedbackOrigin(BaseModel):
    origin: str
    feedbacks: int
