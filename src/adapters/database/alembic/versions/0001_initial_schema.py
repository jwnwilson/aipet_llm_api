"""Initial schema — training_models and training_runs tables as originally created.

Revision ID: 0001
Revises:
Create Date: 2026-05-11
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "training_models",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("base_model", sa.String(255), nullable=False),
        sa.Column("train_data", sa.String(512), nullable=False),
        sa.Column("eval_data", sa.String(512), nullable=False),
        sa.Column("epochs", sa.Integer(), nullable=False),
        sa.Column("patience", sa.Integer(), nullable=False),
        sa.Column("warmup_ratio", sa.Float(), nullable=False),
        sa.Column("remote_backend", sa.String(64), nullable=False),
        sa.Column("skip_generate", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "training_runs",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("model_id", sa.String(36), nullable=False),
        sa.Column("workflow_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("eval_valid_pct", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("training_runs")
    op.drop_table("training_models")
