"""
レジディア分析関連のCRUD操作
"""
from sqlalchemy.orm import Session
from app.models.residia import ResidiaType, ResidiaAnalysis, ResidiaResponseCache
from app.models.session import CounselingSession
from typing import List, Optional, Dict, Tuple
import hashlib
import json
from datetime import datetime

def get_all_residia_types(db: Session) -> List[ResidiaType]:
    """すべてのレジディアタイプを取得"""
    return db.query(ResidiaType).all()

def get_residia_type_by_name(db: Session, name: str) -> Optional[ResidiaType]:
    """名前でレジディアタイプを取得"""
    return db.query(ResidiaType).filter(ResidiaType.name == name).first()

def create_residia_analysis(
    db: Session,
    session_id: int,
    analysis_questions: List[Dict[str, str]],
    primary_type_id: int,
    primary_score: float,
    secondary_type_id: Optional[int] = None,
    secondary_score: Optional[float] = None,
    tertiary_type_id: Optional[int] = None,
    tertiary_score: Optional[float] = None,
    ai_response: str = "",
    response_hash: Optional[str] = None
) -> ResidiaAnalysis:
    """レジディア分析結果を作成"""
    db_analysis = ResidiaAnalysis(
        session_id=session_id,
        analysis_questions=analysis_questions,
        primary_type_id=primary_type_id,
        primary_score=primary_score,
        secondary_type_id=secondary_type_id,
        secondary_score=secondary_score,
        tertiary_type_id=tertiary_type_id,
        tertiary_score=tertiary_score,
        ai_response=ai_response,
        response_hash=response_hash or generate_response_hash(ai_response),
        analysis_count=1
    )
    
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    
    return db_analysis

def get_residia_analysis_by_session(
    db: Session,
    session_id: int
) -> Optional[ResidiaAnalysis]:
    """セッションIDで最新のレジディア分析を取得"""
    return db.query(ResidiaAnalysis).filter(
        ResidiaAnalysis.session_id == session_id
    ).order_by(ResidiaAnalysis.created_at.desc()).first()

def update_residia_analysis(
    db: Session,
    analysis_id: int,
    additional_questions: List[Dict[str, str]],
    ai_response: str
) -> Optional[ResidiaAnalysis]:
    """レジディア分析を更新（マエストロコース用）"""
    db_analysis = db.query(ResidiaAnalysis).filter(
        ResidiaAnalysis.id == analysis_id
    ).first()
    
    if db_analysis:
        # 既存の質問に追加
        all_questions = db_analysis.analysis_questions or []
        all_questions.extend(additional_questions)
        
        db_analysis.analysis_questions = all_questions
        db_analysis.ai_response = ai_response
        db_analysis.response_hash = generate_response_hash(ai_response)
        db_analysis.analysis_count += 1
        db_analysis.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(db_analysis)
    
    return db_analysis

def calculate_residia_scores(
    db: Session,
    session_data: Dict[str, any],
    user_answers: List[Dict[str, str]]
) -> List[Tuple[ResidiaType, float]]:
    """
    セッションデータとユーザー回答からレジディアタイプのスコアを計算
    
    Returns:
        [(ResidiaType, score), ...] のリスト（スコア降順）
    """
    residia_types = get_all_residia_types(db)
    scores = []
    
    # 簡易的なスコアリング（実際の実装ではより洗練されたアルゴリズムを使用）
    for residia_type in residia_types:
        score = 0.0
        
        # キーワードマッチング
        keywords = residia_type.keywords or []
        
        # 初期症状からのマッチング
        initial_prompt = session_data.get("initial_prompt", "").lower()
        for keyword in keywords:
            if keyword.lower() in initial_prompt:
                score += 0.2
        
        # ユーザー回答からのマッチング
        for qa in user_answers:
            answer = qa.get("answer", "").lower()
            for keyword in keywords:
                if keyword.lower() in answer:
                    score += 0.1
        
        # スコアを0.0-1.0に正規化
        score = min(score, 1.0)
        
        if score > 0:
            scores.append((residia_type, score))
    
    # スコア降順でソート
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:3]  # 最大3つまで

def generate_response_hash(response: str) -> str:
    """レスポンスのハッシュを生成（一貫性チェック用）"""
    return hashlib.md5(response.encode()).hexdigest()

def get_or_create_cached_response(
    db: Session,
    input_hash: str,
    generator_func,
    *args,
    **kwargs
) -> str:
    """
    キャッシュされたレスポンスを取得、なければ生成
    70-90%の一貫性を保持
    """
    # キャッシュを検索
    cached = db.query(ResidiaResponseCache).filter(
        ResidiaResponseCache.input_hash == input_hash
    ).first()
    
    if cached:
        # 使用回数を更新
        cached.usage_count += 1
        cached.last_used_at = datetime.utcnow()
        db.commit()
        
        # 70-90%の確率で同じレスポンスを返す
        import random
        if random.random() < 0.8:  # 80%の確率
            return cached.cached_response
    
    # 新しいレスポンスを生成
    new_response = generator_func(*args, **kwargs)
    
    # キャッシュに保存
    if not cached:
        cache_entry = ResidiaResponseCache(
            input_hash=input_hash,
            cached_response=new_response,
            usage_count=1
        )
        db.add(cache_entry)
    else:
        # 20%の確率で新しいレスポンスでキャッシュを更新
        cached.cached_response = new_response
    
    db.commit()
    
    return new_response

def generate_input_hash(
    session_data: Dict[str, any],
    residia_types: List[int],
    user_answers: List[Dict[str, str]]
) -> str:
    """入力データからハッシュを生成"""
    input_data = {
        "initial_prompt": session_data.get("initial_prompt", ""),
        "residia_types": sorted(residia_types),
        "answers": [qa.get("answer", "") for qa in user_answers]
    }
    
    input_string = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(input_string.encode()).hexdigest()