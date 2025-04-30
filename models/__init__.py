# Импортируем всё, чтобы SQLModel.metadata собрал все модели
from .db import engine, init_db, SessionLocal

# metadata = SQLModel.metadata будет содержать таблицы User, Role, UserRole
