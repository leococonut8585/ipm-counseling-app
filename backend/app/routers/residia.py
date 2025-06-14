"""
レジディア分析関連のルーター
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime

from app.database import get_db
from app.crud import residia as crud_residia
from app.crud import session as crud_session
from app.crud import user as crud_user
from app.routers.auth import oauth2_scheme
from app.security import decode_access_token
from app.ai_service import ai_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/residia",
    tags=["レジディア分析"]
)

class ResidiaQuestion(BaseModel):
    """レジディア分析用の質問"""
    question: str
    question_type: str

class ResidiaAnswer(BaseModel):
    """ユーザーの回答 (オブジェクト形式の場合 - 現在は直接使用されていない)"""
    question: Optional[str] = None
    answer: str

class ResidiaAnalysisRequest(BaseModel):
    """レジディア分析リクエスト"""
    session_id: int
    answers: List[str]

class ResidiaTypeInfo(BaseModel):
    """レジディアタイプ情報"""
    id: int
    name: str
    description: str
    score: float
    
    class Config:
        from_attributes = True

class ResidiaAnalysisResponse(BaseModel):
    """レジディア分析レスポンス"""
    id: int
    session_id: int
    primary_type: ResidiaTypeInfo
    secondary_type: Optional[ResidiaTypeInfo]
    tertiary_type: Optional[ResidiaTypeInfo]
    ai_response: str
    analysis_count: int
    can_continue: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="認証情報が無効です", headers={"WWW-Authenticate": "Bearer"})
    email: str = payload.get("sub")
    if email is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="認証情報が無効です", headers={"WWW-Authenticate": "Bearer"})
    user = crud_user.get_user_by_email(db, email=email)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ユーザーが見つかりません")
    return user

@router.get("/types", response_model=List[ResidiaTypeInfo])
async def get_residia_types(db: Session = Depends(get_db)):
    types = crud_residia.get_all_residia_types(db)
    return [ResidiaTypeInfo(id=t.id, name=t.name, description=t.description or "", score=0.0) for t in types]

@router.post("/generate-questions", response_model=List[ResidiaQuestion])
async def generate_residia_questions_endpoint(
    session_id: int, current_user = Depends(get_current_user), db: Session = Depends(get_db)
):
    session = crud_session.get_session(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="セッションが見つかりません")
    
    user_plan_type = getattr(current_user, 'plan_type', 'basic')
    if not hasattr(current_user, 'plan_type'):
        logger.warning(f"User ID {current_user.id} does not have 'plan_type' attribute for residia questions. Defaulting to 'basic'.")

    if user_plan_type == "basic":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="この機能はアドバンスプラン以上で利用可能です")
    
    try:
        session_data = {
            "initial_prompt": session.initial_prompt,
            "physical_diagnosis": session.physical_diagnosis,
            "emotional_diagnosis": session.emotional_diagnosis,
            "unconscious_diagnosis": session.unconscious_diagnosis
        }
        questions_text = await ai_service.generate_residia_questions(session_data=session_data, plan_type=user_plan_type)
        return [ResidiaQuestion(question=q, question_type="initial" if i < 3 else "follow_up") for i, q in enumerate(questions_text)]
    except Exception as e:
        logger.error(f"質問生成エラー for session {session_id}, plan {user_plan_type}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="質問の生成中にエラーが発生しました")

@router.post("/analyze", response_model=ResidiaAnalysisResponse)
async def analyze_residia_endpoint(
    fastapi_request: Request,
    request: ResidiaAnalysisRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        body = await fastapi_request.body()
        logger.info(f"Raw request body for /residia/analyze: {body.decode('utf-8', errors='ignore')}")
    except Exception as e:
        logger.error(f"Error decoding request body: {e}")
    logger.info(f"Parsed ResidiaAnalysisRequest: session_id={request.session_id}, answers={request.answers}")

    session = crud_session.get_session(db, request.session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="セッションが見つかりません")
    
    user_plan_type = getattr(current_user, 'plan_type', 'basic')
    if not hasattr(current_user, 'plan_type'):
        logger.warning(f"User ID {current_user.id} does not have 'plan_type' attribute for residia analysis. Defaulting to 'basic'.")

    if user_plan_type == "basic":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="この機能はアドバンスプラン以上で利用可能です")
    
    existing_analysis = crud_residia.get_residia_analysis_by_session(db, request.session_id)
    if existing_analysis:
        max_count = 1 if user_plan_type == "advance" else 3
        if existing_analysis.analysis_count >= max_count:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"分析回数の上限（{max_count}回）に達しています")
    
    session_data = {
        "initial_prompt": session.initial_prompt,
        "physical_diagnosis": session.physical_diagnosis,
        "emotional_diagnosis": session.emotional_diagnosis,
        "unconscious_diagnosis": session.unconscious_diagnosis
    }

    user_answers_str_list = request.answers

    answers_for_scores = [{"question": f"Q{i+1}", "answer": ans} for i, ans in enumerate(user_answers_str_list)]
    type_scores = crud_residia.calculate_residia_scores(db, session_data, answers_for_scores)
    
    if not type_scores:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="レジディアタイプを判定できませんでした")
    
    primary = type_scores[0] if len(type_scores) > 0 else None
    secondary = type_scores[1] if len(type_scores) > 1 else None
    tertiary = type_scores[2] if len(type_scores) > 2 else None
    
    identified_types = []
    if primary: identified_types.append(primary[0].name)
    if secondary: identified_types.append(secondary[0].name)
    if tertiary: identified_types.append(tertiary[0].name)
    
    try:
        ai_response = await ai_service.analyze_residia(
            session_data=session_data,
            user_answers=user_answers_str_list,
            identified_types=identified_types,
            plan_type=user_plan_type
        )
        
        response_length = len(ai_response)
        if response_length < 3000:
            logger.warning(f"レジディア分析の回答が短すぎます: {response_length}字. Retrying.")
            ai_response = await ai_service.analyze_residia(
                session_data=session_data,
                user_answers=user_answers_str_list,
                identified_types=identified_types,
                plan_type=user_plan_type
            )
            logger.info(f"Residia analysis retry response length: {len(ai_response)}")
        elif response_length > 6000:
            logger.warning(f"レジディア分析の回答が長すぎます: {response_length}字. Truncating.")
            ai_response = ai_response[:6000]
        
        logger.info(f"レジディア分析回答生成完了: {len(ai_response)}字")
        
    except Exception as e:
        logger.error(f"レジディア分析生成エラー: {e}", exc_info=True)
        ai_response = f"申し訳ございません。分析の生成中にエラーが発生しました。"

    answers_for_crud = [{"question": f"UserAnswer{i+1}", "answer": ans} for i, ans in enumerate(user_answers_str_list)]

    if existing_analysis:
        analysis = crud_residia.update_residia_analysis(db, existing_analysis.id, answers_for_crud, ai_response)
    else:
        analysis = crud_residia.create_residia_analysis(
            db, session_id=request.session_id, analysis_questions=answers_for_crud,
            primary_type_id=primary[0].id if primary else None, primary_score=primary[1] if primary else 0.0,
            secondary_type_id=secondary[0].id if secondary else None, secondary_score=secondary[1] if secondary else None,
            tertiary_type_id=tertiary[0].id if tertiary else None, tertiary_score=tertiary[1] if tertiary else None,
            ai_response=ai_response
        )
    
    if not primary: raise HTTPException(status_code=500, detail="Primary residia type could not be determined.")

    max_analysis_count = 1 if user_plan_type == "advance" else 3
    response = ResidiaAnalysisResponse(
        id=analysis.id, session_id=analysis.session_id,
        primary_type=ResidiaTypeInfo(id=primary[0].id, name=primary[0].name, description=primary[0].description or "", score=primary[1]),
        secondary_type=ResidiaTypeInfo(id=secondary[0].id, name=secondary[0].name, description=secondary[0].description or "", score=secondary[1]) if secondary else None,
        tertiary_type=ResidiaTypeInfo(id=tertiary[0].id, name=tertiary[0].name, description=tertiary[0].description or "", score=tertiary[1]) if tertiary else None,
        ai_response=analysis.ai_response, analysis_count=analysis.analysis_count,
        can_continue=analysis.analysis_count < max_analysis_count, created_at=analysis.created_at
    )
    return response

@router.get("/analysis/{session_id}", response_model=ResidiaAnalysisResponse)
async def get_residia_analysis(
    session_id: int, current_user = Depends(get_current_user), db: Session = Depends(get_db)
):
    session = crud_session.get_session(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="セッションが見つかりません")
    
    analysis = crud_residia.get_residia_analysis_by_session(db, session_id)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="レジディア分析が見つかりません")
    
    if not analysis.primary_type:
         raise HTTPException(status_code=500, detail="Primary residia type data missing in analysis record.")

    user_plan_type = getattr(current_user, 'plan_type', 'basic')
    max_analysis_count = 1 if user_plan_type == "advance" else 3

    return ResidiaAnalysisResponse(
        id=analysis.id, session_id=analysis.session_id,
        primary_type=ResidiaTypeInfo.from_attributes(analysis.primary_type),
        secondary_type=ResidiaTypeInfo.from_attributes(analysis.secondary_type) if analysis.secondary_type else None,
        tertiary_type=ResidiaTypeInfo.from_attributes(analysis.tertiary_type) if analysis.tertiary_type else None,
        ai_response=analysis.ai_response, analysis_count=analysis.analysis_count,
        can_continue=analysis.analysis_count < max_analysis_count, created_at=analysis.created_at
    )
    
