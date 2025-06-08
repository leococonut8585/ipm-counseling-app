"""
ユーザー関連のCRUD操作
"""
from sqlalchemy.orm import Session
from app.models.user import User
from app.security import get_password_hash, verify_password
from typing import Optional

def get_user(db: Session, user_id: int) -> Optional[User]:
    """IDでユーザーを取得"""
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """メールアドレスでユーザーを取得"""
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, email: str, password: str, plan_type: str = "basic") -> User:
    """新規ユーザーの作成"""
    # パスワードをハッシュ化
    hashed_password = get_password_hash(password)
    
    # ユーザーオブジェクトの作成
    db_user = User(
        email=email,
        hashed_password=hashed_password,
        plan_type=plan_type
    )
    
    # データベースに追加
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """ユーザーの認証"""
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def update_user_plan(db: Session, user_id: int, plan_type: str) -> Optional[User]:
    """ユーザーのプランを更新"""
    user = get_user(db, user_id)
    if user:
        user.plan_type = plan_type
        db.commit()
        db.refresh(user)
    return user