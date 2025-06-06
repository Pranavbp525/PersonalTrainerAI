"""add_equipment_to_exercise_template

Revision ID: 9f446b6ed3d3
Revises: e5ea20596c44
Create Date: 2025-03-29 19:47:54.875560

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9f446b6ed3d3'
down_revision: Union[str, None] = 'e5ea20596c44'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('exercise_templates', sa.Column('equipment', sa.String(length=50), nullable=False))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('exercise_templates', 'equipment')
    # ### end Alembic commands ###
