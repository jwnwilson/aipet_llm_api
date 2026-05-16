"""Add gguf_path and is_active columns to training_models.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-11
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_models",
        sa.Column("gguf_path", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "training_models",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("training_models", "is_active")
    op.drop_column("training_models", "gguf_path")
