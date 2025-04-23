import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL не задана")

engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_size=10,
    max_overflow=20
)

def init_db() -> None:
    """Создаёт все таблицы из metadata (для локальной разработки)."""
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    """Фабрика сессий для работы с БД."""
    return Session(engine)
