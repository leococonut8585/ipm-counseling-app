"""
ユーザーモデルの定義
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base  # ← ここを変更！

class User(Base):
    """ユーザーテーブルの定義"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    plan_type = Column(String, default="basic")  # basic, advance, maestro
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーション
    sessions = relationship("CounselingSession", back_populates="user")