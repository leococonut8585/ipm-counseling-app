"""
IPMカウンセリングアプリ - メインアプリケーション
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from app.routers import auth, sessions

# 環境変数の読み込み
load_dotenv()

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="IPM Counseling API",
    description="肉体・感情・無意識の三位一体アプローチによるカウンセリングAPI",
    version="1.0.0"
)

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限すること
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルートエンドポイント
@app.get("/")
async def root():
    """APIの稼働確認用エンドポイント"""
    return {
        "message": "IPMカウンセリングAPIへようこそ",
        "status": "稼働中",
        "version": "1.0.0"
    }

# ヘルスチェックエンドポイント
@app.get("/health")
async def health_check():
    """システムの健全性確認用エンドポイント"""
    return {
        "status": "healthy",
        "service": "IPM Counseling API",
        "timestamp": "2024-12-18"
    }

# ルーターの登録
app.include_router(auth.router)
app.include_router(sessions.router)