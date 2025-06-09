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

# AI Model Configuration by Plan
AI_MODEL_BY_PLAN = {
    "basic": {
        "primary": "gemini",
        "fallback_order": ["gemini", "openai", "claude"]
    },
    "advance": {
        "primary": "openai",
        "fallback_order": ["openai", "gemini", "claude"]
    },
    "maestro": {
        "primary": "claude",
        "fallback_order": ["claude", "openai", "gemini"]
    }
}

# デフォルトフォールバック（プランが不明な場合や明示指定時）
DEFAULT_FALLBACK_ORDER = ["claude", "openai", "gemini"]
