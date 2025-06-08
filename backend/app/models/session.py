"""
カウンセリングセッションモデルの定義
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class CounselingSession(Base):
    """カウンセリングセッションテーブル"""
    __tablename__ = "counseling_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # 初期症状・悩み
    initial_prompt = Column(Text, nullable=False)
    
    # 三角形の診断結果
    physical_diagnosis = Column(Text)  # 肉体的な診断
    emotional_diagnosis = Column(Text)  # 感情的な診断
    unconscious_diagnosis = Column(Text)  # 無意識の診断
    
    # カウンセリング回答
    counseling_response = Column(Text)
    
    # AIからの質問と回答（JSON形式で保存）
    ai_questions = Column(JSON)  # [{"question": "...", "answer": "..."}]
    
    # セッションステータス
    status = Column(String, default="in_progress")  # in_progress, completed
    session_count = Column(Integer, default=0)  # カウンセリング継続回数
    
    # ワーク提案
    suggested_work = Column(Text)
    
    # タイムスタンプ
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーション
    user = relationship("User", back_populates="sessions")
# レジディア分析とのリレーション
    residia_analyses = relationship("ResidiaAnalysis", back_populates="session")
    
class IPMKnowledge(Base):
    """IPM知識データベーステーブル"""
    __tablename__ = "ipm_knowledge"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 病気・症状名
    condition_name = Column(String, unique=True, nullable=False, index=True)
    
    # 三要素の説明
    physical_cause = Column(Text)  # 肉体的な要因
    emotional_cause = Column(Text)  # 感情的な要因
    unconscious_cause = Column(Text)  # 無意識の要因
    
    # 関連する病気
    related_conditions = Column(JSON)  # ["病気1", "病気2"]
    
    # キーワード（検索用）
    keywords = Column(JSON)  # ["キーワード1", "キーワード2"]
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)