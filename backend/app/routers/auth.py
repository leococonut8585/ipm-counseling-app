"""
認証関連のルーター
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

router = APIRouter(
    prefix="/auth",
    tags=["認証"]
)

# リクエスト/レスポンスモデル
class UserRegister(BaseModel):
    """ユーザー登録用モデル"""
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    """ログイン用モデル"""
    email: EmailStr
    password: str

class Token(BaseModel):
    """トークンレスポンスモデル"""
    access_token: str
    token_type: str = "bearer"

@router.post("/register", response_model=Token)
async def register(user_data: UserRegister):
    """新規ユーザー登録"""
    # TODO: 実装予定
    return {
        "access_token": "dummy_token",
        "token_type": "bearer"
    }

@router.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    """ユーザーログイン"""
    # TODO: 実装予定
    return {
        "access_token": "dummy_token",
        "token_type": "bearer"
    }