"""
カウンセリングセッション関連のCRUD操作
"""
from sqlalchemy.orm import Session
from app.models.session import CounselingSession, IPMKnowledge
from app.models.user import User
from typing import List, Optional, Dict
import json
from datetime import datetime

def create_session(
    db: Session,
    user_id: int,
    initial_prompt: str
) -> CounselingSession:
    """新規カウンセリングセッションの作成"""
    db_session = CounselingSession(
        user_id=user_id,
        initial_prompt=initial_prompt,
        status="in_progress",
        session_count=0,
        ai_questions=[]
    )
    
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    return db_session

def get_session(db: Session, session_id: int) -> Optional[CounselingSession]:
    """セッションIDでセッションを取得"""
    return db.query(CounselingSession).filter(
        CounselingSession.id == session_id
    ).first()

def get_user_sessions(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 100
) -> List[CounselingSession]:
    """ユーザーのセッション一覧を取得"""
    return db.query(CounselingSession).filter(
        CounselingSession.user_id == user_id
    ).offset(skip).limit(limit).all()

def update_session_diagnosis(
    db: Session,
    session_id: int,
    physical_diagnosis: str,
    emotional_diagnosis: str,
    unconscious_diagnosis: str,
    counseling_response: str
) -> Optional[CounselingSession]:
    """セッションの診断結果を更新"""
    db_session = get_session(db, session_id)
    
    if db_session:
        db_session.physical_diagnosis = physical_diagnosis
        db_session.emotional_diagnosis = emotional_diagnosis
        db_session.unconscious_diagnosis = unconscious_diagnosis
        db_session.counseling_response = counseling_response
        db_session.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(db_session)
    
    return db_session

def add_ai_question_answer(
    db: Session,
    session_id: int,
    question: str,
    answer: str
) -> Optional[CounselingSession]:
    """AIの質問と回答をセッションに追加"""
    db_session = get_session(db, session_id)
    
    if db_session:
        # 既存の質問リストを取得（JSONとして保存されている）
        questions = db_session.ai_questions or []
        
        # 新しい質問と回答を追加
        questions.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # JSONとして保存
        db_session.ai_questions = questions
        db_session.session_count += 1
        db_session.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(db_session)
    
    return db_session

def update_session_work(
    db: Session,
    session_id: int,
    suggested_work: str
) -> Optional[CounselingSession]:
    """セッションのワーク提案を更新"""
    db_session = get_session(db, session_id)
    
    if db_session:
        db_session.suggested_work = suggested_work
        db_session.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(db_session)
    
    return db_session

def complete_session(
    db: Session,
    session_id: int
) -> Optional[CounselingSession]:
    """セッションを完了状態にする"""
    db_session = get_session(db, session_id)
    
    if db_session:
        db_session.status = "completed"
        db_session.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(db_session)
    
    return db_session

# IPM知識データベース関連
def create_ipm_knowledge(
    db: Session,
    condition_name: str,
    physical_cause: str,
    emotional_cause: str,
    unconscious_cause: str,
    related_conditions: List[str] = None,
    keywords: List[str] = None
) -> IPMKnowledge:
    """IPM知識データベースに新規エントリを作成"""
    db_knowledge = IPMKnowledge(
        condition_name=condition_name,
        physical_cause=physical_cause,
        emotional_cause=emotional_cause,
        unconscious_cause=unconscious_cause,
        related_conditions=related_conditions or [],
        keywords=keywords or []
    )
    
    db.add(db_knowledge)
    db.commit()
    db.refresh(db_knowledge)
    
    return db_knowledge

def get_ipm_knowledge_by_condition(
    db: Session,
    condition_name: str
) -> Optional[IPMKnowledge]:
    """病気名でIPM知識を検索"""
    return db.query(IPMKnowledge).filter(
        IPMKnowledge.condition_name == condition_name
    ).first()

def search_ipm_knowledge(
    db: Session,
    keyword: str
) -> List[IPMKnowledge]:
    """キーワードでIPM知識を検索"""
    # JSONフィールド内のキーワード検索は複雑なので、
    # まずは全件取得してPython側でフィルタリング
    all_knowledge = db.query(IPMKnowledge).all()
    
    results = []
    for knowledge in all_knowledge:
        # 病名に含まれるか
        if keyword.lower() in knowledge.condition_name.lower():
            results.append(knowledge)
            continue
        
        # キーワードリストに含まれるか
        if knowledge.keywords:
            for kw in knowledge.keywords:
                if keyword.lower() in kw.lower():
                    results.append(knowledge)
                    break
    
    return results