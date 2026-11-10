"""Initial schema of the tickets database: accounts, users, tickets, messages, customer memory.

The same tables seed.py used to create with create_all(), so existing databases are stamped at this
revision instead of running it (see migrations/common.py).

Revision ID: 0001
Revises:
Create Date: 2026-11-09 22:16:17.927159
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('accounts',
    sa.Column('account_id', sa.String(), nullable=False),
    sa.Column('account_name', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('account_id')
    )
    op.create_table('users',
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('account_id', sa.String(), nullable=False),
    sa.Column('external_user_id', sa.String(), nullable=False),
    sa.Column('user_name', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['account_id'], ['accounts.account_id'], ),
    sa.PrimaryKeyConstraint('user_id'),
    sa.UniqueConstraint('account_id', 'external_user_id', name='uq_user_external_per_account')
    )
    op.create_table('tickets',
    sa.Column('ticket_id', sa.String(), nullable=False),
    sa.Column('account_id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('channel', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['account_id'], ['accounts.account_id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ),
    sa.PrimaryKeyConstraint('ticket_id')
    )
    op.create_table('customer_memory',
    sa.Column('memory_id', sa.String(), nullable=False),
    sa.Column('account_id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('category', sa.String(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('source_ticket_id', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['account_id'], ['accounts.account_id'], ),
    sa.ForeignKeyConstraint(['source_ticket_id'], ['tickets.ticket_id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ),
    sa.PrimaryKeyConstraint('memory_id')
    )
    op.create_table('ticket_messages',
    sa.Column('message_id', sa.String(), nullable=False),
    sa.Column('ticket_id', sa.String(), nullable=False),
    sa.Column('role', sa.Enum('user', 'agent', 'ai', 'system', name='role_enum'), nullable=False),
    sa.Column('content', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.ticket_id'], ),
    sa.PrimaryKeyConstraint('message_id')
    )
    op.create_table('ticket_metadata',
    sa.Column('ticket_id', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('main_issue_type', sa.String(), nullable=True),
    sa.Column('tags', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.ticket_id'], ),
    sa.PrimaryKeyConstraint('ticket_id')
    )


def downgrade() -> None:
    op.drop_table('ticket_metadata')
    op.drop_table('ticket_messages')
    op.drop_table('customer_memory')
    op.drop_table('tickets')
    op.drop_table('users')
    op.drop_table('accounts')
    # Postgres keeps the enum type after its table is dropped
    sa.Enum(name='role_enum').drop(op.get_bind(), checkfirst=True)
