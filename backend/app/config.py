"""
アプリケーション設定
"""
import os
from dotenv import load_dotenv

load_dotenv()

# AI設定
AI_MODEL_PREFERENCE = os.getenv("AI_MODEL_PREFERENCE", "claude")  # claude, openai, gemini
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "300"))  # 5分

# レジディア設定
MIN_RESIDIA_RESPONSE_LENGTH = 3000
MAX_RESIDIA_RESPONSE_LENGTH = 6000

# ログ設定
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")