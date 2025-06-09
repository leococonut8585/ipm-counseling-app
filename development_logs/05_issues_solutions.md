## 2024年12月18日 - レジディア分析422エラーの修正

### 問題
- POST /residia/analyzeで422 Unprocessable Entityエラー
- "Input should be a valid dictionary or object to extract fields from"エラーメッセージ (以前の推測)
- より正確には、FastAPIがリクエストボディを `ResidiaAnalysisRequest` モデルの `answers: List[ResidiaAnswer]` に変換しようとして失敗 (実際には `List[str]` を期待するべき箇所だった)。

### 原因
- `ResidiaAnalysisRequest` モデルの `answers` フィールドが、オブジェクトのリスト (`List[ResidiaAnswer]`) を期待する定義になっていた。
- しかし、クライアントからは文字列のリスト (`List[str]`) として `answers` が送信されていた（またはそのように扱うべきだった）。
- `ai_service.analyze_residia` メソッドは `user_answers: List[str]` を期待するように修正済みだったが、ルーターの入力モデルがそれに追従していなかった。

### 解決策
1. **`ResidiaAnalysisRequest` モデルの修正**: `answers` フィールドの型アノテーションを `List[ResidiaAnswer]` から `List[str]` に変更。
2. **エンドポイント (`analyze_residia_endpoint`) の修正**:
    - Pydanticモデルが `List[str]` を直接受け取るようになったため、エンドポイント内で `request.answers` をそのまま `ai_service.analyze_residia` の `user_answers` パラメータに渡すようにした。以前行っていた `[a.answer for a in request.answers]` のような変換は不要になった。
    - `crud_residia.calculate_residia_scores` や `crud_residia.update_residia_analysis` / `create_residia_analysis` が `List[Dict[str,str]]` 形式の質問リストを期待している可能性があるため、これらのCRUD関数に渡す直前で `user_answers_str_list` (List[str]) を適切な辞書のリスト形式 (`answers_for_scores`, `answers_for_crud`) に変換する処理を追加した。
3. **デバッグログの追加**: `analyze_residia_endpoint` に、受信した生のHTTPリクエストボディと、Pydanticによってパースされたリクエストデータをログに出力する処理を追加し、問題の切り分けを容易にした。

### 学んだこと
- Pydanticモデルの定義は、APIが受け取るリクエストボディの形式と厳密に一致している必要がある。
- 型の不一致は422エラーの一般的な原因である。
- エンドポイントの入力モデルと、それが渡されるサービスレイヤーのメソッドの期待する型は一貫している必要がある。
- 詳細なエラーログとリクエスト内容のロギングは、問題解決において非常に重要である。
- Swagger UI (FastAPIの /docs) でAPIスキーマを確認し、期待されるリクエスト形式を常に把握しておくことが重要。
