from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Строка подключения к вашей базе данных
DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/dlp_system"

# Создание движка SQLAlchemy
engine = create_engine(DATABASE_URL)

# Базовый класс для всех моделей
Base = declarative_base()

# Создание фабрики сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    # Импортируем все модели, чтобы они зарегистрировались в metadata
    from models import user_behavior  # импортируйте тут все модули с таблицами
    Base.metadata.create_all(bind=engine)
