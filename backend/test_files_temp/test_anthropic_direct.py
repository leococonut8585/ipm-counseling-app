import anthropic
import os
from dotenv import load_dotenv

print("=== Anthropic API 直接テスト ===\n")

# .envを読み込む
load_dotenv()

# APIキーを取得
api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"APIキー: {api_key[:15]}...{api_key[-5:]}")
print(f"キーの長さ: {len(api_key)}文字\n")

try:
    # Anthropicクライアントを作成
    client = anthropic.Anthropic(api_key=api_key)
    
    # 簡単なテストメッセージを送信
    print("テストメッセージを送信中...")
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "こんにちは。これはテストメッセージです。"}
        ]
    )
    
    print("✅ 成功！APIキーは有効です！")
    print(f"レスポンス: {response.content[0].text[:50]}...")
    
except anthropic.AuthenticationError as e:
    print("❌ 認証エラー！")
    print(f"エラー詳細: {e}")
    print("\n考えられる原因：")
    print("1. APIキーが無効または期限切れ")
    print("2. APIキーがコピー時に破損")
    print("3. Anthropicアカウントの問題")
    
except Exception as e:
    print(f"❌ その他のエラー: {type(e).__name__}")
    print(f"詳細: {e}")

print("\n=== テスト完了 ===")