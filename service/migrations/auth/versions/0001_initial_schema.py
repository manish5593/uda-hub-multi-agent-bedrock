"""Initial schema of the auth database: customer logins and pending email verification codes.

Revision ID: 0001
Revises:
Create Date: 2026-11-11 01:04:47.243829
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('credentials',
    sa.Column('email', sa.String(length=254), nullable=False),
    sa.Column('external_user_id', sa.String(length=64), nullable=False),
    sa.Column('password_hash', sa.String(length=100), nullable=False),
    sa.Column('failed_logins', sa.Integer(), nullable=False),
    sa.Column('locked_until', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('email'),
    sa.UniqueConstraint('external_user_id')
    )
    op.create_table('verification_codes',
    sa.Column('email', sa.String(length=254), nullable=False),
    sa.Column('code_hash', sa.String(length=64), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('email')
    )


def downgrade() -> None:
    op.drop_table('verification_codes')
    op.drop_table('credentials')
