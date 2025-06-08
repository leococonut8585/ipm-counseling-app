"""
データベース接続設定
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# データベースURL（開発環境ではSQLiteを使用）
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./ipm_counseling.db"  # デフォルトはSQLite
)

# PostgreSQLの場合の処理を追加
if SQLALCHEMY_DATABASE_URL.startswith("postgresql://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg2://"
    )
# SQLite用の接続設定
connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

# エンジンの作成
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args
)

# セッションローカルクラスの作成
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ベースクラス
Base = declarative_base()

# 依存性注入用の関数
def get_db():
    """データベースセッションを取得"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()