"""
カウンセリングセッション関連のルーター
"""
import logging # Added
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.database import get_db
from app.crud import session as crud_session
from app.routers.auth import oauth2_scheme # oauth2_scheme is used by get_current_user
from app.security import decode_access_token
from app.crud.user import get_user_by_email # Used by get_current_user
from app.ai_service import ai_service

logger = logging.getLogger(__name__) # Added

router = APIRouter(
    prefix="/sessions",
    tags=["カウンセリングセッション"]
)

# リクエスト/レスポンスモデル
class SessionCreate(BaseModel):
    """セッション作成用モデル"""
    initial_prompt: str

class DiagnosisUpdate(BaseModel):
    """診断結果更新用モデル"""
    physical_diagnosis: str
    emotional_diagnosis: str
    unconscious_diagnosis: str
    counseling_response: str

class QuestionAnswer(BaseModel):
    """質問と回答のモデル"""
    question: str
    answer: str

class WorkUpdate(BaseModel):
    """ワーク提案更新用モデル"""
    suggested_work: str

class SessionResponse(BaseModel):
    """セッションレスポンスモデル"""
    id: int
    user_id: int
    initial_prompt: str
    physical_diagnosis: Optional[str]
    emotional_diagnosis: Optional[str]
    unconscious_diagnosis: Optional[str]
    counseling_response: Optional[str]
    ai_questions: Optional[List[dict]]
    status: str
    session_count: int
    suggested_work: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# 現在のユーザーを取得する依存関数
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """トークンから現在のユーザーを取得"""
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証情報が無効です",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email: str = payload.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証情報が無効です",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_user_by_email(db, email=email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ユーザーが見つかりません"
        )
    
    return user

@router.post("/", response_model=SessionResponse)
async def create_counseling_session(
    session_data: SessionCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """新規カウンセリングセッションの作成"""
    session = crud_session.create_session(
        db=db,
        user_id=current_user.id,
        initial_prompt=session_data.initial_prompt
    )
    
    return session

@router.get("/", response_model=List[SessionResponse])
async def get_my_sessions(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """自分のセッション一覧を取得"""
    sessions = crud_session.get_user_sessions(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    
    return sessions

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """特定のセッションを取得"""
    session = crud_session.get_session(db, session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="セッションが見つかりません"
        )
    
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="このセッションへのアクセス権限がありません"
        )
    
    return session

@router.put("/{session_id}/diagnosis", response_model=SessionResponse)
async def update_diagnosis(
    session_id: int,
    current_user = Depends(get_current_user), # Assuming current_user has a 'plan_type' attribute
    db: Session = Depends(get_db)
):
    """セッションの診断結果を自動生成して更新"""
    # セッションの所有者確認
    session = crud_session.get_session(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="セッションが見つかりません"
        )

    try:
        # プランベースでAIを選択して診断を生成
        # Ensure current_user has plan_type. This depends on the User model and get_current_user.
        user_plan_type = getattr(current_user, 'plan_type', 'basic') # Default to 'basic' if not present, or handle error
        if not hasattr(current_user, 'plan_type'):
             logger.warning(f"User ID {current_user.id} does not have 'plan_type' attribute. Defaulting to 'basic'. Update User model if plan_type is expected.")


        diagnosis_result = await ai_service.generate_ipm_diagnosis(
            initial_prompt=session.initial_prompt,
            plan_type=user_plan_type  # ユーザープランを渡す
        )

        # 診断に使用されたAIモデルをログに記録（デバッグ用）
        logger.info(f"Diagnosis completed for user plan: {user_plan_type}, session ID: {session_id}")

        # セッションの診断結果を更新
        updated_session = crud_session.update_session_diagnosis(
            db=db,
            session_id=session_id,
            physical_diagnosis=diagnosis_result.get("physical", ""),
            emotional_diagnosis=diagnosis_result.get("emotional", ""),
            unconscious_diagnosis=diagnosis_result.get("unconscious", ""),
            counseling_response=diagnosis_result.get("counseling", "")
        )

        return updated_session

    except Exception as e:
        # Log the full exception details for better debugging
        logger.error(f"AI診断生成エラー for session {session_id}, user plan {user_plan_type}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="診断の生成中にエラーが発生しました"
        )

@router.post("/{session_id}/questions", response_model=SessionResponse)
async def add_question_answer(
    session_id: int,
    qa: QuestionAnswer,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """質問と回答を追加"""
    session = crud_session.get_session(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="セッションが見つかりません"
        )
    
    updated_session = crud_session.add_ai_question_answer(
        db=db,
        session_id=session_id,
        question=qa.question,
        answer=qa.answer
    )
    
    return updated_session