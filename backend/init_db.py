"""
データベース初期化スクリプト
"""
from app.database import engine, Base
# モデルをインポート（これにより、モデルがBaseに登録される）
from app.models.user import User
from app.models.session import CounselingSession, IPMKnowledge

def init_database():
    """データベースのテーブルを作成"""
    print("データベースを初期化しています...")
    
    # 全てのテーブルを作成
    Base.metadata.create_all(bind=engine)
    
    print("データベースの初期化が完了しました！")
    print("作成されたテーブル:")
    print("- users (ユーザー)")
    print("- counseling_sessions (カウンセリングセッション)")
    print("- ipm_knowledge (IPM知識データベース)")

if __name__ == "__main__":
    init_database()