"""empty message

Revision ID: 04d4f05116ed
Revises: 94b198e20946
Create Date: 2024-07-23 01:26:07.117623

"""

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from app.models.base import AESEncryptedColumn


# revision identifiers, used by Alembic.
revision = "04d4f05116ed"
down_revision = "94b198e20946"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "embedding_models",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
        sa.Column(
            "provider", sa.Enum("OPENAI", name="embeddingprovider"), nullable=False
        ),
        sa.Column(
            "model", sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False
        ),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("credentials", AESEncryptedColumn(), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "llms",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
        sa.Column(
            "provider",
            sa.Enum("OPENAI", "GEMINI", "ANTHROPIC_VERTEX", name="llmprovider"),
            nullable=False,
        ),
        sa.Column(
            "model", sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False
        ),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("credentials", AESEncryptedColumn(), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.add_column("chat_engines", sa.Column("llm_id", sa.Integer(), nullable=True))
    op.add_column("chat_engines", sa.Column("fast_llm_id", sa.Integer(), nullable=True))
    op.create_foreign_key(None, "chat_engines", "llms", ["fast_llm_id"], ["id"])
    op.create_foreign_key(None, "chat_engines", "llms", ["llm_id"], ["id"])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, "chat_engines", type_="foreignkey")
    op.drop_constraint(None, "chat_engines", type_="foreignkey")
    op.drop_column("chat_engines", "fast_llm_id")
    op.drop_column("chat_engines", "llm_id")
    op.drop_table("llms")
    op.drop_table("embedding_models")
    # ### end Alembic commands ###
