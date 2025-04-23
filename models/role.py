from typing import Optional
from sqlmodel import SQLModel, Field

class Role(SQLModel, table=True):
    """
    Роль пользователя:
      - id   SERIAL PRIMARY KEY
      - name TEXT NOT NULL UNIQUE
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str         = Field(index=True, unique=True, nullable=False)