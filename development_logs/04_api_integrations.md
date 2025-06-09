## 2025年06月08日 - OpenAI/Gemini API実装

### 完了した作業
1. OpenAI API実装 (`_generate_with_openai`メソッド)
   - OpenAI client (`self.openai_client.chat.completions.create` with `gpt-4`) を使用したAPI呼び出しの実装
   - `asyncio.wait_for`による300秒のタイムアウト処理
   - 応答メッセージからの肉体、感情、無意識、カウンセリングセクションの抽出とパース処理
   - `asyncio.TimeoutError`, `openai.APIError`, 及び一般例外の処理を含むエラーハンドリング
   - `OPENAI_API_KEY`に "dummy" が含まれる場合のモックレスポンス機能

2. Gemini API実装 (`_generate_with_gemini`メソッド)
   - Gemini client (`self.gemini_model.generate_content_async` with `gemini-pro`) を使用したAPI呼び出しの実装
   - `asyncio.wait_for`による300秒のタイムアウト処理
   - 応答メッセージからの肉体、感情、無意識、カウンセリングセクションの抽出とパース処理
   - `asyncio.TimeoutError`, `google.api_core.exceptions.GoogleAPIError`, 及び一般例外の処理を含むエラーハンドリング
   - `GOOGLE_API_KEY`に "dummy" が含まれる場合、またはGeminiクライアントが未初期化の場合のモックレスポンス機能

### 技術的な決定事項
- OpenAI: `gpt-4` モデルを利用。エラーハンドリングは `openai.APIError` をキャッチ。
- Gemini: `gemini-pro` モデルを利用。エラーハンドリングは `google.api_core.exceptions.GoogleAPIError` をキャッチ。
- 共通: Claudeの実装 (`_generate_with_claude`) を参考に、レスポンスのパースロジック（セクションキーワードによる分割と整形）を統一的に適用。
- 共通: APIキーに "dummy" が含まれている場合、またはクライアントが適切に初期化されていない場合は、実際のAPIコールを行わずモックデータを返すことで、開発中のテストを容易化。
- 共通: ログ出力には標準の `logging` モジュールを使用し、エラー発生時には詳細を記録。

### 次のステップ
- 各APIの個別テスト実施
- 統合テストの準備
## 2024年12月18日 - AI連携の完全リファクタリング

### 実施した改善
1. **プロンプトの構造化**
   - マーカーベースの明確な出力形式を定義
   - system_promptとuser_promptの役割を明確化
   - 文字数制限を明示

2. **統一パース処理の実装**
   - ResponseParserクラスによる一元的なパース処理
   - マーカーベースの抽出とフォールバック処理
   - レスポンス検証機能の追加

3. **エラーハンドリングの強化**
   - API固有のエラー処理
   - リトライロジックの実装
   - フォールバック機能の追加

4. **デバッグ機能の追加**
   - 詳細なログ出力
   - 生のレスポンス記録
   - パース結果の検証

### テスト結果
- Claude: [結果を記載]
- OpenAI: [結果を記載]
- Gemini: [結果を記載]

### 次のステップ
- レジディア分析機能の同様の改善
- パフォーマンス最適化
- キャッシング機能の検討
```
