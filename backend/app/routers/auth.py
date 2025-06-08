"""
認証関連のルーター
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import Optional

from app.database import get_db
from app.crud import user as crud_user
from app.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(
    prefix="/auth",
    tags=["認証"]
)

# OAuth2設定
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# リクエスト/レスポンスモデル
class UserRegister(BaseModel):
    """ユーザー登録用モデル"""
    email: EmailStr
    password: str
    plan_type: Optional[str] = "basic"

class UserResponse(BaseModel):
    """ユーザーレスポンスモデル"""
    id: int
    email: str
    plan_type: str
    is_active: bool
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    """トークンレスポンスモデル"""
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """トークンデータモデル"""
    email: Optional[str] = None

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserRegister,
    db: Session = Depends(get_db)
):
    """新規ユーザー登録"""
    # メールアドレスの重複チェック
    db_user = crud_user.get_user_by_email(db, email=user_data.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="このメールアドレスは既に登録されています"
        )
    
    # ユーザーの作成
    user = crud_user.create_user(
        db=db,
        email=user_data.email,
        password=user_data.password,
        plan_type=user_data.plan_type
    )
    
    return user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """ユーザーログイン"""
    # ユーザーの認証
    user = crud_user.authenticate_user(
        db=db,
        email=form_data.username,  # OAuth2ではusernameフィールドを使用
        password=form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="メールアドレスまたはパスワードが正しくありません",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # トークンの作成
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """現在のユーザー情報を取得"""
    # この実装は後で追加します
    # まずは基本的な登録・ログインを動作確認しましょう
    pass