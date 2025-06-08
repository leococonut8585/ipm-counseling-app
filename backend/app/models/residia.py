"""
レジディア（幼少期のトラウマ）関連のモデル定義
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class ResidiaType(Base):
    """レジディアタイプマスターテーブル"""
    __tablename__ = "residia_types"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)  # 背信、不道徳、無欲、哀感、苛烈、切断
    description = Column(Text)  # タイプの説明
    keywords = Column(JSON)  # 判定用キーワード
    file_path = Column(String)  # ODTファイルのパス
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ResidiaAnalysis(Base):
    """レジディア分析結果テーブル"""
    __tablename__ = "residia_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("counseling_sessions.id"), nullable=False)
    
    # 分析質問と回答
    analysis_questions = Column(JSON)  # [{"question": "...", "answer": "..."}]
    
    # 判定結果（最大3つ）
    primary_type_id = Column(Integer, ForeignKey("residia_types.id"))
    primary_score = Column(Float)  # 該当度スコア（0.0-1.0）
    
    secondary_type_id = Column(Integer, ForeignKey("residia_types.id"))
    secondary_score = Column(Float)
    
    tertiary_type_id = Column(Integer, ForeignKey("residia_types.id"))
    tertiary_score = Column(Float)
    
    # AIの回答
    ai_response = Column(Text)  # 3000-6000字の回答
    response_hash = Column(String)  # 回答の一貫性確保用ハッシュ
    
    # 分析回数（マエストロコースで使用）
    analysis_count = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーション
    session = relationship("CounselingSession", back_populates="residia_analyses")
    primary_type = relationship("ResidiaType", foreign_keys=[primary_type_id])
    secondary_type = relationship("ResidiaType", foreign_keys=[secondary_type_id])
    tertiary_type = relationship("ResidiaType", foreign_keys=[tertiary_type_id])

class ResidiaResponseCache(Base):
    """レジディア回答キャッシュテーブル（一貫性保持用）"""
    __tablename__ = "residia_response_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 入力情報のハッシュ（症状+質問回答+レジディアタイプの組み合わせ）
    input_hash = Column(String, unique=True, index=True)
    
    # キャッシュされた回答
    cached_response = Column(Text)
    
    # 使用回数（人気度の把握）
    usage_count = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, default=datetime.utcnow)