"""
AI連携サービス
Claude, OpenAI, Geminiとの統合
"""
import os
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv
import asyncio
import anthropic
import openai
from google import generativeai as genai
import logging
from app.file_manager import get_file_content_for_ai

# 環境変数の読み込み
load_dotenv()

# ログ設定
logger = logging.getLogger(__name__)

# API設定
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# タイムアウト設定（長文処理用）
AI_TIMEOUT = 300  # 5分

class AIService:
    """AI統合サービスクラス"""
    
    def __init__(self):
        """APIクライアントの初期化"""
        # Claude
        if ANTHROPIC_API_KEY:
            self.claude_client = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY
            )
        else:
            logger.warning("Anthropic APIキーが設定されていません")
            self.claude_client = None
        
        # OpenAI
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            self.openai_client = openai
        else:
            logger.warning("OpenAI APIキーが設定されていません")
            self.openai_client = None
        
        # Gemini
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        else:
            logger.warning("Google APIキーが設定されていません")
            self.gemini_model = None
    
    async def generate_ipm_diagnosis(
        self,
        initial_prompt: str,
        ai_model: Literal["claude", "openai", "gemini"] = "claude"
    ) -> Dict[str, str]:
        """
        IPM診断を生成（肉体・感情・無意識の三要素）
        
        Returns:
            {
                "physical": "肉体的な診断",
                "emotional": "感情的な診断",
                "unconscious": "無意識の診断",
                "counseling": "カウンセリング回答"
            }
        """
        # IPMファイル情報を取得
        ipm_files = get_file_content_for_ai("ipm")
        
        system_prompt = f"""
あなたはIPM（統合心身医学）の専門家です。
以下のIPM知識ファイルを参照して、ユーザーの症状を肉体・感情・無意識の三要素から分析してください。

参照ファイル：
{ipm_files['instructions']}
ファイルパス：{', '.join(ipm_files['file_paths'])}

これらのファイルの内容を完全に理解し、それに基づいて診断を行ってください。
"""
        
        user_prompt = f"""
以下の症状/悩みについて、IPMの観点から診断してください：

{initial_prompt}

以下の形式で回答してください：
1. 肉体的な要因（200-300字）
2. 感情的な要因（200-300字）
3. 無意識の要因（200-300字）
4. 総合的なカウンセリング（1000-2000字）
"""
        
        if ai_model == "claude" and self.claude_client:
            return await self._generate_with_claude(system_prompt, user_prompt)
        elif ai_model == "openai" and self.openai_client:
            return await self._generate_with_openai(system_prompt, user_prompt)
        elif ai_model == "gemini" and self.gemini_model:
            return await self._generate_with_gemini(system_prompt, user_prompt)
        else:
            raise ValueError(f"利用可能なAIモデルがありません: {ai_model}")
    
    async def generate_residia_questions(
        self,
        session_data: Dict[str, any],
        ai_model: Literal["claude", "openai", "gemini"] = "claude"
    ) -> List[str]:
        """
        レジディア分析用の質問を生成（5問）
        """
        # レジディアファイル情報を取得
        residia_files = get_file_content_for_ai("residia")
        
        system_prompt = f"""
あなたはレジディア（幼少期のトラウマ）分析の専門家です。
以下の6つのレジディアタイプのファイルを参照してください。

参照ファイル：
{residia_files['instructions']}
ファイルパス：{', '.join(residia_files['file_paths'])}

これらのファイルの内容を完全に理解し、ユーザーの状態から最適なレジディアタイプを判定するための質問を生成してください。
"""
        
        user_prompt = f"""
ユーザーの初期症状：
{session_data.get('initial_prompt', '')}

IPM診断結果：
- 肉体的診断：{session_data.get('physical_diagnosis', '')}
- 感情的診断：{session_data.get('emotional_diagnosis', '')}
- 無意識の診断：{session_data.get('unconscious_diagnosis', '')}

この情報を元に、レジディアタイプを特定するための質問を5つ生成してください。
質問は、幼少期の経験や現在の行動パターンの関連を探るものにしてください。
"""
        
        if ai_model == "claude" and self.claude_client:
            questions = await self._generate_questions_with_claude(system_prompt, user_prompt)
        elif ai_model == "openai" and self.openai_client:
            questions = await self._generate_questions_with_openai(system_prompt, user_prompt)
        elif ai_model == "gemini" and self.gemini_model:
            questions = await self._generate_questions_with_gemini(system_prompt, user_prompt)
        else:
            raise ValueError(f"利用可能なAIモデルがありません: {ai_model}")
        
        return questions[:5]  # 最大5問
    
    async def analyze_residia(
        self,
        session_data: Dict[str, any],
        user_answers: List[Dict[str, str]],
        identified_types: List[str],
        ai_model: Literal["claude", "openai", "gemini"] = "claude"
    ) -> str:
        """
        レジディア分析結果を生成（3000-6000字）
        """
        # 特定されたレジディアタイプのファイル情報を取得
        residia_responses = []
        for residia_type in identified_types[:3]:  # 最大3タイプ
            file_info = get_file_content_for_ai("residia", residia_type)
            residia_responses.append(f"{residia_type}: {file_info['file_paths'][0] if file_info['file_paths'] else 'ファイルなし'}")
        
        system_prompt = f"""
あなたは深い共感力を持つカウンセラーです。
レジディア（幼少期のトラウマ）の観点から、愛情と寄り添いに満ちた回答を提供してください。

参照するレジディアファイル：
{chr(10).join(residia_responses)}

これらのファイルの内容を完全に理解し、ユーザーの幼少期の経験が現在にどう影響しているかを深く洞察してください。
"""
        
        user_prompt = f"""
ユーザーの初期症状：
{session_data.get('initial_prompt', '')}

レジディア分析の質問と回答：
{chr(10).join([f"Q: {qa['question']}{chr(10)}A: {qa['answer']}" for qa in user_answers])}

判定されたレジディアタイプ：
{', '.join(identified_types[:3])}

上記の情報を元に、3000字以上6000字未満で、愛情と寄り添いに満ちたカウンセリング回答を作成してください。
ユーザーの幼少期の経験を否定せず、それが現在の強さにつながっていることを伝えてください。
"""
        
        if ai_model == "claude" and self.claude_client:
            return await self._generate_residia_analysis_with_claude(system_prompt, user_prompt)
        elif ai_model == "openai" and self.openai_client:
            return await self._generate_residia_analysis_with_openai(system_prompt, user_prompt)
        elif ai_model == "gemini" and self.gemini_model:
            return await self._generate_residia_analysis_with_gemini(system_prompt, user_prompt)
        else:
            raise ValueError(f"利用可能なAIモデルがありません: {ai_model}")
    
    # Claude実装
async def _generate_with_claude(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        """Claude APIでIPM診断を生成"""
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.claude_client.messages.create,
                    model="claude-3-opus-20240229",
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                ),
                timeout=AI_TIMEOUT
            )
            
            # レスポンスをパース
            content = response.content[0].text
            
            # セクションごとに分割
            result = {
                "physical": "",
                "emotional": "",
                "unconscious": "",
                "counseling": ""
            }
            
            # 各セクションを抽出
            sections = content.split('\n\n')
            current_section = None
            
            for section in sections:
                if '肉体的な要因' in section or '肉体的要因' in section:
                    current_section = 'physical'
                elif '感情的な要因' in section or '感情的要因' in section:
                    current_section = 'emotional'
                elif '無意識の要因' in section or '無意識的要因' in section:
                    current_section = 'unconscious'
                elif '総合的なカウンセリング' in section or 'カウンセリング' in section:
                    current_section = 'counseling'
                elif current_section:
                    # 現在のセクションにテキストを追加
                    result[current_section] += section + '\n'
            
            # 各セクションのテキストをクリーンアップ
            for key in result:
                result[key] = result[key].strip()
                # セクションタイトルを除去
                for title in ['肉体的な要因', '感情的な要因', '無意識の要因', '総合的なカウンセリング', '1.', '2.', '3.', '4.']:
                    result[key] = result[key].replace(title, '').strip()
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("Claude APIがタイムアウトしました")
            raise
        except Exception as e:
            logger.error(f"Claude APIエラー: {e}")
            raise
    
    async def _generate_questions_with_claude(self, system_prompt: str, user_prompt: str) -> List[str]:
        """Claude APIで質問を生成"""
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.claude_client.messages.create,
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                ),
                timeout=AI_TIMEOUT
            )
            
            content = response.content[0].text
            
            # 質問を抽出（改行で分割）
            questions = [q.strip() for q in content.split('\n') if q.strip() and not q.strip().startswith('#')]
            
            return questions[:5]
            
        except Exception as e:
            logger.error(f"Claude API質問生成エラー: {e}")
            raise
    
    async def _generate_residia_analysis_with_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Claude APIでレジディア分析を生成"""
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.claude_client.messages.create,
                    model="claude-3-opus-20240229",
                    max_tokens=8000,  # 6000字対応
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                ),
                timeout=AI_TIMEOUT
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude APIレジディア分析エラー: {e}")
            raise
    
    # OpenAIとGeminiの実装は省略（基本的に同じ構造）
    # 実際の実装では、各APIの特性に合わせて調整が必要
    
# シングルトンインスタンス
ai_service = AIService()