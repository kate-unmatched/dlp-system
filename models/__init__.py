# Импортируем всё, чтобы SQLModel.metadata собрал все модели
from .db import engine, init_db, get_session
from .user import User
from .role import Role
from .user_role import UserRole

# metadata = SQLModel.metadata будет содержать таблицы User, Role, UserRole
