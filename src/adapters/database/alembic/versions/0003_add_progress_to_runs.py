"""Add progress and progress_detail columns to training_runs.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-12
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("training_runs", sa.Column("progress", sa.Float(), nullable=True))
    op.add_column("training_runs", sa.Column("progress_detail", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "progress_detail")
    op.drop_column("training_runs", "progress")
