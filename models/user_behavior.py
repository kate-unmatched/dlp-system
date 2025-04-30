from sqlalchemy import Column, Integer, String, DateTime
import datetime
from models.db import Base

class UserBehavior(Base):
    __tablename__ = "user_behavior"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    event_time = Column(DateTime, default=datetime.datetime.utcnow)
    feature_vector = Column(String, nullable=False)
    danger_level = Column(Integer, nullable=False)
