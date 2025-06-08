"""
レジディア分析関連のルーター
"""
from fastapi import APIRouter, Depends, HTTPException, status
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
from app.file_manager import get_file_content_for_ai

router = APIRouter(
    prefix="/residia",
    tags=["レジディア分析"]
)

# リクエスト/レスポンスモデル
class ResidiaQuestion(BaseModel):
    """レジディア分析用の質問"""
    question: str
    question_type: str  # "initial" or "follow_up"

class ResidiaAnswer(BaseModel):
    """ユーザーの回答"""
    question: str
    answer: str

class ResidiaAnalysisRequest(BaseModel):
    """レジディア分析リクエスト"""
    session_id: int
    answers: List[ResidiaAnswer]

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
    can_continue: bool  # さらに分析を続けられるか
    created_at: datetime
    
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
    
    user = crud_user.get_user_by_email(db, email=email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ユーザーが見つかりません"
        )
    
    return user

@router.get("/types", response_model=List[ResidiaTypeInfo])
async def get_residia_types(
    db: Session = Depends(get_db)
):
    """すべてのレジディアタイプを取得"""
    types = crud_residia.get_all_residia_types(db)
    
    return [
        ResidiaTypeInfo(
            id=t.id,
            name=t.name,
            description=t.description or "",
            score=0.0  # スコアは分析時に計算
        )
        for t in types
    ]

@router.post("/generate-questions", response_model=List[ResidiaQuestion])
async def generate_residia_questions(
    session_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """レジディア分析用の質問を生成"""
    # セッションの所有者確認
    session = crud_session.get_session(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="セッションが見つかりません"
        )
    
    # プランチェック（アドバンス以上）
    if current_user.plan_type == "basic":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="この機能はアドバンスプラン以上で利用可能です"
        )
    
    # TODO: AIを使って質問を生成
    # ここでは仮の質問を返す
    questions = [
        ResidiaQuestion(
            question="幼少期に最も強く記憶に残っている出来事は何ですか？",
            question_type="initial"
        ),
        ResidiaQuestion(
            question="両親や養育者との関係で、今でも心に残っている言葉はありますか？",
            question_type="initial"
        ),
        ResidiaQuestion(
            question="子供の頃、自分の感情を素直に表現できましたか？できなかった場合、その理由は？",
            question_type="initial"
        ),
        ResidiaQuestion(
            question="過去の経験で、今の自分の行動パターンに影響を与えていると感じるものはありますか？",
            question_type="initial"
        ),
        ResidiaQuestion(
            question="信頼していた人に裏切られたと感じた経験はありますか？",
            question_type="initial"
        )
    ]
    
    return questions

@router.post("/analyze", response_model=ResidiaAnalysisResponse)
async def analyze_residia(
    request: ResidiaAnalysisRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """レジディア分析を実行"""
    # セッションの所有者確認
    session = crud_session.get_session(db, request.session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="セッションが見つかりません"
        )
    
    # プランチェック
    if current_user.plan_type == "basic":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="この機能はアドバンスプラン以上で利用可能です"
        )
    
    # 既存の分析を確認
    existing_analysis = crud_residia.get_residia_analysis_by_session(
        db, request.session_id
    )
    
    # 分析回数制限チェック
    if existing_analysis:
        max_count = 1 if current_user.plan_type == "advance" else 3
        if existing_analysis.analysis_count >= max_count:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"分析回数の上限（{max_count}回）に達しています"
            )
    
    # セッションデータを準備
    session_data = {
        "initial_prompt": session.initial_prompt,
        "physical_diagnosis": session.physical_diagnosis,
        "emotional_diagnosis": session.emotional_diagnosis,
        "unconscious_diagnosis": session.unconscious_diagnosis
    }
    
    # 回答データを準備
    answers = [{"question": a.question, "answer": a.answer} for a in request.answers]
    
    # レジディアタイプのスコアを計算
    type_scores = crud_residia.calculate_residia_scores(
        db, session_data, answers
    )
    
    if not type_scores:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="レジディアタイプを判定できませんでした"
        )
    
    # 最大3つのタイプを取得
    primary = type_scores[0] if len(type_scores) > 0 else None
    secondary = type_scores[1] if len(type_scores) > 1 else None
    tertiary = type_scores[2] if len(type_scores) > 2 else None
    
    # TODO: AIを使って3000-6000字の回答を生成
    # ここでは仮の回答を生成
    ai_response = f"""
あなたの回答から、{primary[0].name}のレジディアが最も強く影響していることがわかりました。

{primary[0].name}のレジディアは、{primary[0].description}

このタイプの特徴として...

（実際の実装では、ここでAIがファイルを参照して3000-6000字の詳細な分析を生成します）

あなたの幼少期の経験は、現在のあなたの行動パターンや感情反応に深く影響を与えています。
しかし、それは決してあなたの価値を損なうものではありません。
むしろ、その経験があったからこそ、今のあなたの強さと優しさがあるのです。

これからの人生において、このレジディアとどのように向き合っていくか...
"""
    
    # 分析結果を保存
    if existing_analysis:
        # 既存の分析を更新
        analysis = crud_residia.update_residia_analysis(
            db,
            existing_analysis.id,
            answers,
            ai_response
        )
    else:
        # 新規作成
        analysis = crud_residia.create_residia_analysis(
            db,
            session_id=request.session_id,
            analysis_questions=answers,
            primary_type_id=primary[0].id,
            primary_score=primary[1],
            secondary_type_id=secondary[0].id if secondary else None,
            secondary_score=secondary[1] if secondary else None,
            tertiary_type_id=tertiary[0].id if tertiary else None,
            tertiary_score=tertiary[1] if tertiary else None,
            ai_response=ai_response
        )
    
    # レスポンスを構築
    max_count = 1 if current_user.plan_type == "advance" else 3
    
    response = ResidiaAnalysisResponse(
        id=analysis.id,
        session_id=analysis.session_id,
        primary_type=ResidiaTypeInfo(
            id=primary[0].id,
            name=primary[0].name,
            description=primary[0].description or "",
            score=primary[1]
        ),
        secondary_type=ResidiaTypeInfo(
            id=secondary[0].id,
            name=secondary[0].name,
            description=secondary[0].description or "",
            score=secondary[1]
        ) if secondary else None,
        tertiary_type=ResidiaTypeInfo(
            id=tertiary[0].id,
            name=tertiary[0].name,
            description=tertiary[0].description or "",
            score=tertiary[1]
        ) if tertiary else None,
        ai_response=analysis.ai_response,
        analysis_count=analysis.analysis_count,
        can_continue=analysis.analysis_count < max_count,
        created_at=analysis.created_at
    )
    
    return response

@router.get("/analysis/{session_id}", response_model=ResidiaAnalysisResponse)
async def get_residia_analysis(
    session_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """セッションのレジディア分析結果を取得"""
    # セッションの所有者確認
    session = crud_session.get_session(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="セッションが見つかりません"
        )
    
    analysis = crud_residia.get_residia_analysis_by_session(db, session_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="レジディア分析が見つかりません"
        )
    
    # レスポンスを構築
    max_count = 1 if current_user.plan_type == "advance" else 3
    
    response = ResidiaAnalysisResponse(
        id=analysis.id,
        session_id=analysis.session_id,
        primary_type=ResidiaTypeInfo(
            id=analysis.primary_type.id,
            name=analysis.primary_type.name,
            description=analysis.primary_type.description or "",
            score=analysis.primary_score
        ),
        secondary_type=ResidiaTypeInfo(
            id=analysis.secondary_type.id,
            name=analysis.secondary_type.name,
            description=analysis.secondary_type.description or "",
            score=analysis.secondary_score
        ) if analysis.secondary_type else None,
        tertiary_type=ResidiaTypeInfo(
            id=analysis.tertiary_type.id,
            name=analysis.tertiary_type.name,
            description=analysis.tertiary_type.description or "",
            score=analysis.tertiary_score
        ) if analysis.tertiary_type else None,
        ai_response=analysis.ai_response,
        analysis_count=analysis.analysis_count,
        can_continue=analysis.analysis_count < max_count,
        created_at=analysis.created_at
    )
    
    return response