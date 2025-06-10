import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("=== 環境変数デバッグ開始 ===\n")

# 現在のディレクトリ
print(f"現在のディレクトリ: {os.getcwd()}")
print(f"Pythonパス: {sys.executable}")
print(f"Python version: {sys.version}\n")

# .envファイルの検索
env_locations = [
    Path(".env"),
    Path("../.env"),
    Path("backend/.env"),
    Path("app/.env"),
]

print("=== .envファイルの検索 ===")
for loc in env_locations:
    abs_path = Path(os.getcwd()) / loc
    exists = abs_path.exists()
    print(f"{loc}: {'存在する' if exists else '存在しない'} ({abs_path})")
    if exists:
        print(f"  -> ファイルサイズ: {abs_path.stat().st_size} bytes")

# .envを読み込む
print("\n=== load_dotenv実行 ===")
result = load_dotenv(verbose=True)
print(f"load_dotenv結果: {result}")

# APIキーの確認
print("\n=== APIキーの確認 ===")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

def check_key(name, key):
    if not key:
        print(f"{name}: 未設定！")
    else:
        print(f"{name}: 設定済み")
        print(f"  長さ: {len(key)}文字")
        print(f"  最初の15文字: {key[:15]}...")
        print(f"  最後の5文字: ...{key[-5:]}")
        # ダブルクォートチェック
        if key.startswith('"') or key.endswith('"'):
            print(f"  ⚠️ 警告: ダブルクォートが含まれています！")
        # スペースチェック
        if key != key.strip():
            print(f"  ⚠️ 警告: 前後に空白が含まれています！")

check_key("ANTHROPIC_API_KEY", anthropic_key)
check_key("OPENAI_API_KEY", openai_key)
check_key("GOOGLE_API_KEY", google_key)

# 生の環境変数も確認
print("\n=== 生の環境変数（os.environ）===")
for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
    if key in os.environ:
        print(f"{key}: 存在する（{len(os.environ[key])}文字）")
    else:
        print(f"{key}: 存在しない")