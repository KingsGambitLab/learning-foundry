"""Initial schema — 12 legacy tables ported from SQLiteWorkflowStore.

Revision ID: 0001
Revises:
Create Date: 2026-05-14
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workflow_runs",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("workflow_runs_updated_at_idx", "workflow_runs", ["updated_at"])

    op.create_table(
        "workflow_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("workflow_events_run_id_idx", "workflow_events", ["run_id", "sequence_no"])

    op.create_table(
        "course_runs",
        sa.Column("course_run_id", sa.Text(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("package_type", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("course_runs_updated_at_idx", "course_runs", ["updated_at"])

    op.create_table(
        "course_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("course_events_run_id_idx", "course_events", ["course_run_id", "sequence_no"])

    op.create_table(
        "learner_enrollments",
        sa.Column("enrollment_id", sa.Text(), primary_key=True),
        sa.Column("learner_id", sa.Text(), nullable=False),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_enrollments_learner_idx", "learner_enrollments", ["learner_id"])
    op.create_index("learner_enrollments_course_idx", "learner_enrollments", ["course_run_id"])

    op.create_table(
        "learner_submissions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("enrollment_id", sa.Text(), nullable=False),
        sa.Column("deliverable_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_submissions_enrollment_idx", "learner_submissions", ["enrollment_id", "created_at"])

    op.create_table(
        "learner_workspace_sessions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("enrollment_id", sa.Text(), nullable=False),
        sa.Column("deliverable_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_workspace_sessions_enrollment_idx", "learner_workspace_sessions", ["enrollment_id", "created_at"])

    op.create_table(
        "publish_snapshots",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("publish_snapshots_course_idx", "publish_snapshots", ["course_run_id", "created_at"])

    op.create_table(
        "creator_feedback",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("creator_feedback_course_idx", "creator_feedback", ["course_run_id", "created_at"])

    op.create_table(
        "learner_feedback",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("enrollment_id", sa.Text(), nullable=False),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_feedback_enrollment_idx", "learner_feedback", ["enrollment_id", "created_at"])

    op.create_table(
        "learner_eval_reports",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("enrollment_id", sa.Text(), nullable=False),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("publish_snapshot_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_eval_reports_enrollment_idx", "learner_eval_reports", ["enrollment_id", "created_at"])

    op.create_table(
        "creator_assets",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("creator_assets_course_idx", "creator_assets", ["course_run_id", "updated_at"])


def downgrade() -> None:
    for table in [
        "creator_assets",
        "learner_eval_reports",
        "learner_feedback",
        "creator_feedback",
        "publish_snapshots",
        "learner_workspace_sessions",
        "learner_submissions",
        "learner_enrollments",
        "course_events",
        "course_runs",
        "workflow_events",
        "workflow_runs",
    ]:
        op.drop_table(table)
