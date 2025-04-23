from typing import Optional
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    """
    Пользователь:
      - id              SERIAL PRIMARY KEY
      - username        TEXT NOT NULL UNIQUE
      - hashed_password TEXT NOT NULL
      - is_active       BOOLEAN NOT NULL DEFAULT TRUE
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str       = Field(index=True, unique=True, nullable=False)
    hashed_password: str= Field(nullable=False)
    is_active: bool     = Field(default=True, nullable=False)
