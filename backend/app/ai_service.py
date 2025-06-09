import os
import re
import asyncio
import logging
from typing import Dict, List, Optional, Literal # Ensure Optional is imported
from dotenv import load_dotenv

import anthropic
import openai
from google import generativeai as genai
from google.api_core import exceptions as google_exceptions

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Configure basic logging

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# AI Timeout
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", 300))  # Default to 300 seconds (5 minutes)

# Custom Error
class InsufficientPlanError(Exception):
    """プランが不十分な場合のエラー"""
    pass

# Plan validation function
def validate_plan_for_ai(plan_type: str, requested_ai: str) -> bool:
    """プランが要求されたAIを使用可能か検証"""
    if plan_type == "basic" and requested_ai == "claude":
        return False
    return True

class ResponseParser:
    """AIレスポンスの統一パーサー"""

    @staticmethod
    def parse_ipm_diagnosis(content: str) -> Dict[str, str]:
        """IPM診断レスポンスをパース"""
        result = {
            "physical": "",
            "emotional": "",
            "unconscious": "",
            "counseling": ""
        }

        patterns = {
            'physical': r'###PHYSICAL_START###\s*(.*?)\s*###PHYSICAL_END###',
            'emotional': r'###EMOTIONAL_START###\s*(.*?)\s*###EMOTIONAL_END###',
            'unconscious': r'###UNCONSCIOUS_START###\s*(.*?)\s*###UNCONSCIOUS_END###',
            'counseling': r'###COUNSELING_START###\s*(.*?)\s*###COUNSELING_END###'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
            else:
                logger.warning(f"Marker not found for {key} in IPM diagnosis. Attempting fallback.")
                result[key] = ResponseParser._fallback_extract(content, key)

        filled_sections = sum(1 for v in result.values() if v)
        if filled_sections < 2:
            logger.warning(f"IPM diagnosis parsing result is insufficient. Filled sections: {filled_sections}")
            logger.debug(f"Raw content for IPM diagnosis (first 500 chars): {content[:500]}")

        return result

    @staticmethod
    def _fallback_extract(content: str, section_type: str) -> str:
        """マーカーが見つからない場合のフォールバック抽出 for IPM"""
        keywords = {
            'physical': ['肉体的な要因', '肉体的要因', '肉体的', '身体的', '体の', 'Physical'],
            'emotional': ['感情的な要因', '感情的要因', '感情的', '情緒的', '気持ち', 'Emotional'],
            'unconscious': ['無意識の要因', '無意識的要因', '無意識', '潜在意識', '深層心理', 'Unconscious'],
            'counseling': ['総合的なカウンセリング', 'カウンセリング', '総合的', 'アドバイス', 'Counseling']
        }

        for keyword in keywords.get(section_type, []):
            pattern = rf'{re.escape(keyword)}.*?(?:[:：]|\n)\s*(.*?)(?=\n\n|\n(?:###[A-Z_]+_START###|1\.|2\.|3\.|4\.)|$)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_text = match.group(1).strip()
                if "###" not in extracted_text:
                    return extracted_text

        logger.debug(f"Fallback extraction failed for section: {section_type}")
        return ""

    @staticmethod
    def parse_residia_questions(content: str) -> List[str]:
        """レジディア質問レスポンスをパース"""
        questions = []

        for i in range(1, 6):
            pattern = rf'###Q{i}_START###\s*(.*?)\s*###Q{i}_END###'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                questions.append(match.group(1).strip())

        if len(questions) < 3:
            logger.warning("Marker-based extraction for Residia questions yielded less than 3 questions. Attempting fallback.")
            fallback_pattern = r'(?:^|\n)\s*(?:(?:\d+\.?|\*|-)\s+)?(.*?)(?=\n\s*(?:(?:\d+\.?|\*|-)\s+)|$|###Q\d_START###)'
            fallback_matches = re.findall(fallback_pattern, content, re.MULTILINE)

            extracted_fallback_questions = []
            for match_text in fallback_matches:
                cleaned_q = match_text.strip()
                if cleaned_q and len(cleaned_q) > 10 and not cleaned_q.startswith("###") and not cleaned_q.endswith("###"):
                    extracted_fallback_questions.append(cleaned_q)

            if questions:
                for fq in extracted_fallback_questions:
                    if fq not in questions and len(questions) < 5:
                        questions.append(fq)
            else:
                 questions = extracted_fallback_questions

        return questions[:5]

class AIService:
    def __init__(self):
        self.claude_client = None
        if ANTHROPIC_API_KEY and "dummy" not in ANTHROPIC_API_KEY:
            self.claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            logger.warning("Anthropic API key not available or is a dummy key.")

        self.openai_client = None
        if OPENAI_API_KEY and "dummy" not in OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        else:
            logger.warning("OpenAI API key not available or is a dummy key.")

        self.gemini_model = None
        if GOOGLE_API_KEY and "dummy" not in GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")
        else:
            logger.warning("Google API key not available or is a dummy key.")

    def get_ai_models_for_plan(self, plan_type: str) -> tuple[str, list[str]]:
        """
        プランに基づくAIモデルとフォールバック順序を取得
        """
        from app.config import AI_MODEL_BY_PLAN, DEFAULT_FALLBACK_ORDER

        if plan_type in AI_MODEL_BY_PLAN:
            config = AI_MODEL_BY_PLAN[plan_type]
            primary = config.get("primary")
            order = config.get("fallback_order")
            if not primary or not order:
                logger.error(f"Plan configuration for '{plan_type}' is malformed. Using default fallback order.")
                return DEFAULT_FALLBACK_ORDER[0], DEFAULT_FALLBACK_ORDER
            return primary, order
        else:
            logger.warning(f"Unknown plan type: '{plan_type}', using default fallback order.")
            return DEFAULT_FALLBACK_ORDER[0], DEFAULT_FALLBACK_ORDER

    def _create_ipm_diagnosis_prompt(self, initial_prompt: str) -> tuple[str, str]:
        system_prompt = """あなたはIPM（統合心身医学）の専門家です。
ユーザーの症状を肉体・感情・無意識の三要素から分析し、必ず以下の形式で回答してください。

回答形式：
- 各セクションは必ず指定されたマーカーで開始する
- 各セクションは独立した段落として記載する
- 文字数は厳密に守る
- セクションの順序を変更しない"""
        
        user_prompt = f"""以下の症状について分析してください：

{initial_prompt}

必ず以下の形式で回答してください：

###PHYSICAL_START###
（ここに200-300字で肉体的な要因の分析を記載。症状の身体的側面、生活習慣、環境要因など）
###PHYSICAL_END###

###EMOTIONAL_START###
（ここに200-300字で感情的な要因の分析を記載。ストレス、人間関係、感情パターンなど）
###EMOTIONAL_END###

###UNCONSCIOUS_START###
（ここに200-300字で無意識の要因の分析を記載。深層心理、過去の経験、潜在的な信念など）
###UNCONSCIOUS_END###

###COUNSELING_START###
（ここに1000-2000字で総合的なカウンセリングを記載。三要素を統合した理解と具体的なアドバイス）
###COUNSELING_END###"""
        return system_prompt, user_prompt

    def _create_residia_questions_prompt(self, session_data: dict) -> tuple[str, str]:
        system_prompt = """あなたはレジディア（幼少期のトラウマ）分析の専門家です。
6つのレジディアタイプを理解し、ユーザーの状態から最適な質問を生成します。

質問生成のルール：
- 幼少期の体験と現在の行動パターンの関連を探る
- 非侵襲的で共感的な表現を使う
- 具体的な記憶よりも感覚や感情を重視する"""
        
        user_prompt = f"""以下の情報から、レジディアタイプを特定する質問を5つ生成してください：

初期症状：{session_data.get('initial_prompt', '')}
肉体的診断：{session_data.get('physical_diagnosis', '')}
感情的診断：{session_data.get('emotional_diagnosis', '')}
無意識の診断：{session_data.get('unconscious_diagnosis', '')}

質問は以下の形式で出力してください：

###Q1_START###
（1つ目の質問）
###Q1_END###

###Q2_START###
（2つ目の質問）
###Q2_END###

###Q3_START###
（3つ目の質問）
###Q3_END###

###Q4_START###
（4つ目の質問）
###Q4_END###

###Q5_START###
（5つ目の質問）
###Q5_END###"""
        return system_prompt, user_prompt

    def _create_mock_response(self, ai_type: str, type: str = "ipm") -> Dict[str, str] | List[str]:
        logger.info(f"Creating mock response for {ai_type} ({type})")
        if type == "ipm":
            return {
                "physical": f"【{ai_type} Mock】###PHYSICAL_START###\n肉体的な要因として、ストレスによる自律神経の乱れが考えられます。特に交感神経が過度に優位になることで、頭痛や身体の重さを引き起こしている可能性があります。規則正しい生活リズムと適度な運動が改善に効果的でしょう。\n###PHYSICAL_END###",
                "emotional": f"【{ai_type} Mock】###EMOTIONAL_START###\n感情面では、仕事のプレッシャーに対する不安や焦燥感が蓄積されています。完璧主義的な傾向や、他者からの評価を過度に気にする傾向が、ストレスを増幅させている可能性があります。\n###EMOTIONAL_END###",
                "unconscious": f"【{ai_type} Mock】###UNCONSCIOUS_START###\n無意識レベルでは、幼少期の体験から形成された「頑張らなければ認められない」という信念が影響している可能性があります。この深層心理的なパターンが、現在のストレス反応を強化しています。\n###UNCONSCIOUS_END###",
                "counseling": f"【{ai_type} Mock】###COUNSELING_START###\nあなたの症状は、肉体・感情・無意識の三つのレベルが複雑に絡み合って生じています。まず肉体面では、十分な睡眠と栄養バランスの取れた食事を心がけ、週に2-3回の軽い運動を取り入れることをお勧めします。感情面では、ストレスの原因を具体的に書き出し、優先順位をつけて対処することが有効です。また、完璧を求めすぎず、「70%でも十分」という考え方を取り入れてみてください。無意識のレベルでは、自己肯定感を高めるワークが重要です。毎日寝る前に、その日の小さな成功体験を3つ書き出すことから始めてみましょう。これらの統合的なアプローチにより、徐々に症状の改善が期待できます。\n###COUNSELING_END###"
            }
        elif type == "residia_questions":
            return [
                f"【{ai_type} Mock】###Q1_START###\n幼少期に、ありのままの自分でいることを許されなかったと感じる経験はありましたか？\n###Q1_END###",
                f"【{ai_type} Mock】###Q2_START###\n何かを達成したり、誰かの期待に応えたりしないと価値がないと感じることはありますか？\n###Q2_END###",
                f"【{ai_type} Mock】###Q3_START###\n自分の感情やニーズを表現することをためらったり、難しいと感じたりすることはありますか？\n###Q3_END###",
                f"【{ai_type} Mock】###Q4_START###\n人間関係において、見捨てられることへの強い不安を感じたり、過度に相手に合わせようとしたりする傾向はありますか？\n###Q4_END###",
                f"【{ai_type} Mock】###Q5_START###\n完璧主義的な傾向や、自分自身に厳しすぎる傾向はありますか？\n###Q5_END###",
            ]
        return {}

    async def _generate_with_claude(self, system_prompt: str, user_prompt: str, is_ipm: bool) -> Dict[str, str] | List[str]:
        if not self.claude_client:
            logger.warning("Claude client not available. Returning mock response.")
            mock_type = "ipm" if is_ipm else "residia_questions"
            if is_ipm:
                 return self._create_mock_response("Claude", "ipm") # type: ignore
            else:
                 return self._create_mock_response("Claude", "residia_questions") # type: ignore

        try:
            logger.info("Calling Claude API")
            logger.debug(f"Claude System prompt (first 200 chars): {system_prompt[:200]}...")
            logger.debug(f"Claude User prompt (first 200 chars): {user_prompt[:200]}...")

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.claude_client.messages.create,
                    model="claude-3-opus-20240229",
                    max_tokens=4000,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                ),
                timeout=AI_TIMEOUT
            )
            
            content = response.content[0].text
            logger.debug(f"Claude raw response (first 1000 chars): {content[:1000]}")

            if is_ipm:
                if "###PHYSICAL_START###" in content:
                    return ResponseParser.parse_ipm_diagnosis(content)
                else:
                    logger.warning("Claude response for IPM does not contain new markers, using legacy parser.")
                    return self._legacy_parse_claude_response(content)
            else:
                return ResponseParser.parse_residia_questions(content)

        except asyncio.TimeoutError:
            logger.error("Claude API call timed out.")
            raise
        except anthropic.RateLimitError as e:
            logger.error(f"Claude API rate limit exceeded: {e}")
            raise
        except anthropic.AuthenticationError as e:
            logger.error(f"Claude API authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Claude API unexpected error: {type(e).__name__}: {e}")
            raise

    def _legacy_parse_claude_response(self, content: str) -> Dict[str, str]:
        """旧形式のClaude応答用パーサー（後方互換性 for IPM）"""
        logger.info("Using legacy Claude parser for IPM diagnosis.")
        result = {"physical": "", "emotional": "", "unconscious": "", "counseling": ""}

        physical_match = re.search(r"(?:肉体的な要因|肉体的要因|1\.)[:：\s]*(.*?)(?=\n\n(?:感情的な要因|感情的要因|2\.)|$)", content, re.DOTALL | re.IGNORECASE)
        if physical_match: result['physical'] = physical_match.group(1).strip()

        emotional_match = re.search(r"(?:感情的な要因|感情的要因|2\.)[:：\s]*(.*?)(?=\n\n(?:無意識の要因|無意識的要因|3\.)|$)", content, re.DOTALL | re.IGNORECASE)
        if emotional_match: result['emotional'] = emotional_match.group(1).strip()

        unconscious_match = re.search(r"(?:無意識の要因|無意識的要因|3\.)[:：\s]*(.*?)(?=\n\n(?:総合的なカウンセリング|カウンセリング|4\.)|$)", content, re.DOTALL | re.IGNORECASE)
        if unconscious_match: result['unconscious'] = unconscious_match.group(1).strip()

        counseling_match = re.search(r"(?:総合的なカウンセリング|カウンセリング|4\.)[:：\s]*(.*)", content, re.DOTALL | re.IGNORECASE)
        if counseling_match: result['counseling'] = counseling_match.group(1).strip()

        if not any(result.values()) and content:
            parts = content.split('\n\n', 3)
            keys = ['physical', 'emotional', 'unconscious', 'counseling']
            for i, part_content in enumerate(parts):
                if i < len(keys):
                    result[keys[i]] = part_content.strip()

        filled_sections = sum(1 for v in result.values() if v)
        logger.debug(f"Legacy Claude parser filled {filled_sections} sections for IPM.")
        return result

    async def _generate_with_openai(self, system_prompt: str, user_prompt: str, is_ipm: bool) -> Dict[str, str] | List[str]:
        if not self.openai_client:
            logger.warning("OpenAI client not available. Returning mock response.")
            mock_type = "ipm" if is_ipm else "residia_questions"
            if is_ipm:
                 return self._create_mock_response("OpenAI", "ipm") # type: ignore
            else:
                 return self._create_mock_response("OpenAI", "residia_questions") # type: ignore

        try:
            logger.info("Calling OpenAI API")
            logger.debug(f"OpenAI System prompt (first 200 chars): {system_prompt[:200]}...")
            logger.debug(f"OpenAI User prompt (first 200 chars): {user_prompt[:200]}...")

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                ),
                timeout=AI_TIMEOUT
            )

            content = response.choices[0].message.content
            if content is None:
                logger.error("OpenAI API returned None content.")
                return {} if is_ipm else [] # type: ignore
            logger.debug(f"OpenAI raw response (first 1000 chars): {content[:1000]}")

            return ResponseParser.parse_ipm_diagnosis(content) if is_ipm else ResponseParser.parse_residia_questions(content)

        except asyncio.TimeoutError:
            logger.error("OpenAI API call timed out.")
            raise
        except openai.RateLimitError as e:
            logger.error(f"OpenAI API rate limit exceeded: {e}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI API authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API unexpected error: {type(e).__name__}: {e}")
            raise

    async def _generate_with_gemini(self, system_prompt: str, user_prompt: str, is_ipm: bool) -> Dict[str, str] | List[str]:
        if not self.gemini_model:
            logger.warning("Gemini model not available. Returning mock response.")
            mock_type = "ipm" if is_ipm else "residia_questions"
            if is_ipm:
                 return self._create_mock_response("Gemini", "ipm") # type: ignore
            else:
                 return self._create_mock_response("Gemini", "residia_questions") # type: ignore

        try:
            logger.info("Calling Gemini API")
            effective_prompt = f"{system_prompt}\n\n{user_prompt}"
            logger.debug(f"Gemini Combined prompt (first 200 chars): {effective_prompt[:200]}...")

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.gemini_model.generate_content,
                    effective_prompt
                ),
                timeout=AI_TIMEOUT
            )

            content = response.text if hasattr(response, 'text') else ''
            if not content and hasattr(response, 'parts') and response.parts:
                 content = "".join(part.text for part in response.parts if hasattr(part, 'text'))

            if not content:
                logger.error("Gemini API returned empty content.")
                return {} if is_ipm else [] # type: ignore

            logger.debug(f"Gemini raw response (first 1000 chars): {content[:1000]}")

            return ResponseParser.parse_ipm_diagnosis(content) if is_ipm else ResponseParser.parse_residia_questions(content)

        except asyncio.TimeoutError:
            logger.error("Gemini API call timed out.")
            raise
        except google_exceptions.RetryError as e:
            logger.error(f"Gemini API retry error: {e}")
            raise
        except google_exceptions.InvalidArgumentError as e:
            logger.error(f"Gemini API invalid argument: {e}")
            raise
        except Exception as e:
            logger.error(f"Gemini API unexpected error: {type(e).__name__}: {e}")
            raise

    async def _fallback_to_available_ai(self, system_prompt: str, user_prompt: str, is_ipm: bool, preferred_order: List[str]) -> Dict[str, str] | List[str]:
        logger.info(f"Attempting fallback for {'IPM' if is_ipm else 'Residia questions'} with order: {preferred_order}")
        model_map = {
            "claude": (self._generate_with_claude, self.claude_client),
            "openai": (self._generate_with_openai, self.openai_client),
            "gemini": (self._generate_with_gemini, self.gemini_model)
        }
        for model_name in preferred_order:
            if model_name not in model_map:
                logger.warning(f"Unknown model '{model_name}' in fallback order. Skipping.")
                continue
            method, client = model_map[model_name]
            if client:
                logger.info(f"Fallback: Trying {model_name}.")
                try:
                    return await method(system_prompt, user_prompt, is_ipm)
                except Exception as e:
                    logger.warning(f"Fallback to {model_name} failed: {type(e).__name__}: {e}")
            else:
                logger.debug(f"Fallback: {model_name} client not available.")
        logger.error(f"All AI fallbacks in order {preferred_order} failed or no clients available.")
        mock_type_str = "ipm" if is_ipm else "residia_questions"
        logger.warning(f"Returning emergency mock response for {mock_type_str} from _fallback_to_available_ai.")
        if is_ipm:
            return self._create_mock_response("EmergencyFallbackInMethod", "ipm") # type: ignore
        else:
            return self._create_mock_response("EmergencyFallbackInMethod", "residia_questions") # type: ignore

    def _validate_ipm_response(self, response: Optional[Dict[str, str]]) -> bool:
        if not response:
            logger.warning("Validation failed: Response is None or empty.")
            return False
        required_keys = ["physical", "emotional", "unconscious", "counseling"]
        if not all(key in response for key in required_keys):
            logger.warning(f"Validation failed: Missing keys. Found: {list(response.keys())}, Expected: {required_keys}")
            return False
        filled_count = sum(1 for key in required_keys if response.get(key, "").strip())
        if filled_count < 2:
            logger.warning(f"Validation failed: Insufficient content. Filled sections: {filled_count}/{len(required_keys)}")
            return False
        if not response.get("counseling", "").strip():
            logger.warning("Validation failed: Counseling section is empty, which is mandatory.")
            return False
        logger.info("IPM response validation successful.")
        return True

    async def generate_ipm_diagnosis(
        self,
        initial_prompt: str,
        ai_model: Optional[str] = None,
        plan_type: Optional[str] = None
    ) -> Dict[str, str]:
        from app.config import DEFAULT_FALLBACK_ORDER

        primary_model_to_use: Optional[str] = None
        current_fallback_order: List[str] = []

        if ai_model:
            logger.info(f"Explicit AI model specified: {ai_model}. Plan-based selection will be overridden for primary choice.")
            primary_model_to_use = ai_model
            if ai_model in DEFAULT_FALLBACK_ORDER:
                temp_order = [m for m in DEFAULT_FALLBACK_ORDER if m != ai_model]
                current_fallback_order = [ai_model] + temp_order
            else:
                current_fallback_order = [ai_model] + [m for m in DEFAULT_FALLBACK_ORDER if m != ai_model]
            logger.info(f"Using fallback order for explicit AI: {current_fallback_order}")
        elif plan_type:
            _primary, _fallback_order = self.get_ai_models_for_plan(plan_type)
            primary_model_to_use = _primary
            current_fallback_order = _fallback_order
            logger.info(f"Plan-based AI selection for plan '{plan_type}'. Primary: {primary_model_to_use}, Order: {current_fallback_order}")
        else:
            raise ValueError("Either ai_model or plan_type must be specified for IPM diagnosis")

        system_prompt, user_prompt = self._create_ipm_diagnosis_prompt(initial_prompt)
        last_error: Optional[Exception] = None
        result: Optional[Dict[str, str]] = None

        for model_name_in_order in current_fallback_order:
            try:
                logger.info(f"Attempting IPM diagnosis with {model_name_in_order} (Plan: {plan_type or 'N/A'}, Explicit: {ai_model or 'N/A'})")

                if model_name_in_order == "claude" and self.claude_client:
                    result = await self._generate_with_claude(system_prompt, user_prompt, is_ipm=True) # type: ignore
                elif model_name_in_order == "openai" and self.openai_client:
                    result = await self._generate_with_openai(system_prompt, user_prompt, is_ipm=True) # type: ignore
                elif model_name_in_order == "gemini" and self.gemini_model:
                    result = await self._generate_with_gemini(system_prompt, user_prompt, is_ipm=True) # type: ignore
                else:
                    logger.warning(f"Client for {model_name_in_order} not available or model unknown. Trying next in fallback order.")
                    if not last_error:
                        last_error = Exception(f"Client for {model_name_in_order} not available.")
                    continue

                if self._validate_ipm_response(result):
                    logger.info(f"IPM diagnosis successful with {model_name_in_order}.")
                    return result # type: ignore
                else:
                    logger.warning(f"Response validation failed for {model_name_in_order}. Content: {str(result)[:200]}...")
                    last_error = Exception(f"Response validation failed for {model_name_in_order}.")
            except Exception as e:
                logger.error(f"Error during IPM diagnosis with {model_name_in_order}: {type(e).__name__} - {e}")
                last_error = e

        logger.error(f"All AI models in fallback order {current_fallback_order} failed to provide a valid IPM diagnosis. Last error: {last_error}")
        if isinstance(last_error, Exception):
            raise last_error
        elif last_error is not None:
             raise Exception(str(last_error))
        else:
            raise Exception("IPM diagnosis failed for an unknown reason after trying all fallbacks.")

    async def generate_residia_questions(
        self,
        session_data: Dict[str, any],
        ai_model: Optional[str] = None,
        plan_type: Optional[str] = None
    ) -> List[str]:
        from app.config import DEFAULT_FALLBACK_ORDER

        primary_model_to_use: Optional[str] = None
        current_fallback_order: List[str] = []

        if ai_model:
            logger.info(f"Explicit AI model specified for Residia questions: {ai_model}.")
            primary_model_to_use = ai_model
            if ai_model in DEFAULT_FALLBACK_ORDER:
                temp_order = [m for m in DEFAULT_FALLBACK_ORDER if m != ai_model]
                current_fallback_order = [ai_model] + temp_order
            else:
                current_fallback_order = [ai_model] + [m for m in DEFAULT_FALLBACK_ORDER if m != ai_model]
            logger.info(f"Using fallback order for explicit AI (Residia): {current_fallback_order}")
        elif plan_type:
            _primary, _fallback_order = self.get_ai_models_for_plan(plan_type)
            primary_model_to_use = _primary
            current_fallback_order = _fallback_order
            logger.info(f"Plan-based AI for Residia questions (Plan '{plan_type}'). Primary: {primary_model_to_use}, Order: {current_fallback_order}")
        else:
            raise ValueError("Either ai_model or plan_type must be specified for Residia questions")

        system_prompt, user_prompt = self._create_residia_questions_prompt(session_data)
        last_error: Optional[Exception] = None
        questions: Optional[List[str]] = None

        for model_name_in_order in current_fallback_order:
            try:
                logger.info(f"Attempting Residia questions with {model_name_in_order} (Plan: {plan_type or 'N/A'}, Explicit: {ai_model or 'N/A'})")
                if model_name_in_order == "claude" and self.claude_client:
                    questions = await self._generate_with_claude(system_prompt, user_prompt, is_ipm=False) # type: ignore
                elif model_name_in_order == "openai" and self.openai_client:
                    questions = await self._generate_with_openai(system_prompt, user_prompt, is_ipm=False) # type: ignore
                elif model_name_in_order == "gemini" and self.gemini_model:
                    questions = await self._generate_with_gemini(system_prompt, user_prompt, is_ipm=False) # type: ignore
                else:
                    logger.warning(f"Client for {model_name_in_order} (Residia questions) not available or model unknown.")
                    if not last_error: last_error = Exception(f"Client for {model_name_in_order} not available.")
                    continue

                if questions and len(questions) >= 3:
                    logger.info(f"Residia questions successful with {model_name_in_order} (Count: {len(questions)}).")
                    return questions[:5]
                else:
                    logger.warning(f"Insufficient questions from {model_name_in_order} for Residia. Count: {len(questions) if questions else 'None'}.")
                    last_error = Exception(f"Insufficient questions from {model_name_in_order}.")
            except Exception as e:
                logger.error(f"Error with {model_name_in_order} for Residia questions: {type(e).__name__} - {e}")
                last_error = e

        logger.error(f"All AI models for Residia questions failed. Last error: {last_error}. Order: {current_fallback_order}")
        if self._create_mock_response:
             return self._create_mock_response("CriticalFallbackMock", "residia_questions") # type: ignore
        raise last_error if isinstance(last_error, Exception) else Exception("Residia questions failed.")


    async def analyze_residia(
        self,
        session_data: Dict[str, any],
        user_answers: List[Dict[str, str]],
        identified_types: List[str],
        ai_model: Optional[str] = None,
        plan_type: Optional[str] = None
    ) -> str:
        from app.config import DEFAULT_FALLBACK_ORDER
        logger.info(f"analyze_residia called. Plan: {plan_type}, Explicit AI: {ai_model}")

        primary_model_to_use: Optional[str] = None
        current_fallback_order: List[str] = []

        if ai_model:
            logger.info(f"Explicit AI model specified for Residia analysis: {ai_model}.")
            primary_model_to_use = ai_model
            if ai_model in DEFAULT_FALLBACK_ORDER:
                temp_order = [m for m in DEFAULT_FALLBACK_ORDER if m != ai_model]
                current_fallback_order = [ai_model] + temp_order
            else:
                current_fallback_order = [ai_model] + [m for m in DEFAULT_FALLBACK_ORDER if m != ai_model]
        elif plan_type:
            _primary, _fallback_order = self.get_ai_models_for_plan(plan_type)
            primary_model_to_use = _primary
            current_fallback_order = _fallback_order
        else:
            logger.warning("Neither plan_type nor ai_model specified for analyze_residia. Using default fallback.")
            primary_model_to_use = DEFAULT_FALLBACK_ORDER[0]
            current_fallback_order = DEFAULT_FALLBACK_ORDER

        logger.info(f"Residia Analysis - Primary: {primary_model_to_use}, Order: {current_fallback_order}")

        _system_prompt = f"Analyze Residia for types: {', '.join(identified_types)} based on provided data. Use {primary_model_to_use} logic."
        _user_prompt = f"Session: {session_data.get('initial_prompt', '')}. Answers: {user_answers}. Analyze."
        chosen_model_for_mock = primary_model_to_use or current_fallback_order[0]

        logger.warning(f"analyze_residia is using mock logic. Chosen AI for mock: {chosen_model_for_mock}")

        return f"Mock analysis for plan '{plan_type}' (tried {chosen_model_for_mock} first based on logic) for types: {', '.join(identified_types)}. User prompt: {session_data.get('initial_prompt', '')}. This function's core generation logic needs full implementation."

ai_service = AIService()
