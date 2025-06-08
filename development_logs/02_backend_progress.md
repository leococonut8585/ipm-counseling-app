### Git管理の改善
- __pycache__ファイルがステージングに含まれる問題を発見・解決
- .gitignoreを強化して、サブディレクトリの__pycache__も確実に除外
- README.mdにプロジェクト概要を追加
## 2024年12月18日 - データベース構造の実装

### 完了した作業
1. SQLAlchemy接続設定（database.py）
   - SQLiteをデフォルトDBとして設定
   - セッション管理機能の実装
2. モデル定義
   - Userモデル：ユーザー管理（email, password, plan_type）
   - CounselingSessionモデル：セッション記録（症状、診断、AIとの対話）
   - IPMKnowledgeモデル：知識DB（病気と三要素の関係）
3. データベース初期化
   - init_db.pyスクリプト作成
   - 全テーブルの作成成功
   - ipm_counseling.dbファイル生成確認

### 解決した問題
- PostgreSQL接続エラー → SQLiteに変更して解決
- 外部キー参照エラー → インポート順序の修正で解決
- __pycache__のGit混入 → git resetとファイル個別追加で解決

### 作成したテーブル構造
- users: ユーザー情報管理
- counseling_sessions: カウンセリング履歴
- ipm_knowledge: IPM理論に基づく知識DB

### 次のステップ
- CRUD操作の実装（Create, Read, Update, Delete）
- ユーザー認証機能の実装（JWT）
- セッション管理APIの作成