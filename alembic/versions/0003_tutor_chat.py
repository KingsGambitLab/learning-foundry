"""Tutor chat persistence: tutor_chat_messages.

Lab-tutor transcripts were browser-localStorage only (lost on
clear-site-data / different device / STORAGE_VERSION bump). Persist
them server-side keyed by (user_id, session_id); localStorage stays as
an offline cache.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-16
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tutor_chat_messages",
        sa.Column("message_id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index(
        "tutor_chat_messages_idx",
        "tutor_chat_messages",
        ["user_id", "session_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("tutor_chat_messages_idx", table_name="tutor_chat_messages")
    op.drop_table("tutor_chat_messages")
