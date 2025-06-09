import os
import re
import asyncio
import logging
from typing import Dict, List, Optional, Literal
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
                # Fallback: try legacy method
                logger.warning(f"Marker not found for {key} in IPM diagnosis. Attempting fallback.")
                result[key] = ResponseParser._fallback_extract(content, key)

        filled_sections = sum(1 for v in result.values() if v)
        if filled_sections < 2: # Expect at least 2 sections, ideally all 4
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

        # Try keyword-based extraction (simplified)
        # This fallback is basic and might need more sophisticated logic depending on actual legacy formats
        for keyword in keywords.get(section_type, []):
            # Attempt to find "Keyword:[ ]Content" or "Keyword\nContent"
            # This regex is an example and may need refinement
            pattern = rf'{re.escape(keyword)}.*?(?:[:：]|\n)\s*(.*?)(?=\n\n|\n(?:###[A-Z_]+_START###|1\.|2\.|3\.|4\.)|$)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_text = match.group(1).strip()
                # Avoid re-extracting other sections' content if they accidentally match
                if "###" not in extracted_text: # Simple check to avoid grabbing marked sections
                    return extracted_text

        logger.debug(f"Fallback extraction failed for section: {section_type}")
        return ""

    @staticmethod
    def parse_residia_questions(content: str) -> List[str]:
        """レジディア質問レスポンスをパース"""
        questions = []

        for i in range(1, 6): # Q1 to Q5
            pattern = rf'###Q{i}_START###\s*(.*?)\s*###Q{i}_END###'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                questions.append(match.group(1).strip())

        if len(questions) < 3: # If not enough questions found with markers
            logger.warning("Marker-based extraction for Residia questions yielded less than 3 questions. Attempting fallback.")
            # Fallback: Numbered list or bullet points
            # Regex looks for lines starting with a number/bullet, capturing the text after.
            fallback_pattern = r'(?:^|\n)\s*(?:(?:\d+\.?|\*|-)\s+)?(.*?)(?=\n\s*(?:(?:\d+\.?|\*|-)\s+)|$|###Q\d_START###)'
            fallback_matches = re.findall(fallback_pattern, content, re.MULTILINE)

            extracted_fallback_questions = []
            for match_text in fallback_matches:
                cleaned_q = match_text.strip()
                # Basic filter: not too short, not a marker itself
                if cleaned_q and len(cleaned_q) > 10 and not cleaned_q.startswith("###") and not cleaned_q.endswith("###"):
                    extracted_fallback_questions.append(cleaned_q)

            if questions: # If some questions were found by markers, append fallback ones if different
                for fq in extracted_fallback_questions:
                    if fq not in questions and len(questions) < 5:
                        questions.append(fq)
            else: # No marker questions found, use fallback directly
                 questions = extracted_fallback_questions

        return questions[:5] # Max 5 questions

class AIService:
    def __init__(self):
        self.claude_client = None
        if ANTHROPIC_API_KEY and "dummy" not in ANTHROPIC_API_KEY:
            self.claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            logger.warning("Anthropic API key not available or is a dummy key.")

        self.openai_client = None
        if OPENAI_API_KEY and "dummy" not in OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) # Correct instantiation
        else:
            logger.warning("OpenAI API key not available or is a dummy key.")

        self.gemini_model = None
        if GOOGLE_API_KEY and "dummy" not in GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Using a common model
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")
        else:
            logger.warning("Google API key not available or is a dummy key.")

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
        return {} # Should not happen with current logic


    async def _generate_with_claude(self, system_prompt: str, user_prompt: str, is_ipm: bool) -> Dict[str, str] | List[str]:
        if not self.claude_client:
            logger.warning("Claude client not available. Returning mock response.")
            # Ensure the type matches the expected return type based on is_ipm
            mock_type = "ipm" if is_ipm else "residia_questions"
            return self._create_mock_response("Claude", mock_type) # type: ignore

        try:
            logger.info("Calling Claude API")
            logger.debug(f"Claude System prompt (first 200 chars): {system_prompt[:200]}...")
            logger.debug(f"Claude User prompt (first 200 chars): {user_prompt[:200]}...")

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.claude_client.messages.create,
                    model="claude-3-opus-20240229", # specified model
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
                if "###PHYSICAL_START###" in content: # Check for new markers
                    return ResponseParser.parse_ipm_diagnosis(content)
                else:
                    logger.warning("Claude response for IPM does not contain new markers, using legacy parser.")
                    return self._legacy_parse_claude_response(content)
            else: # Residia questions
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
            return self._create_mock_response("OpenAI", mock_type) # type: ignore

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
                # Return a typed empty response or raise, depending on strictness
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
            return self._create_mock_response("Gemini", mock_type) # type: ignore

        try:
            logger.info("Calling Gemini API")
            # Gemini's `generate_content` can take `system_instruction` in the model or `contents`
            # For `gemini-1.5-flash` (and similar models), it's best to use `GenerativeModel(model_name, system_instruction=...)`
            # or `start_chat(system_instruction=...)`.
            # If system prompt must be dynamic per call and not using chat, combining is an option.
            # Let's assume `self.gemini_model` is initialized without a global system_instruction.

            effective_prompt = f"{system_prompt}\n\n{user_prompt}"
            logger.debug(f"Gemini Combined prompt (first 200 chars): {effective_prompt[:200]}...")

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.gemini_model.generate_content,
                    effective_prompt,
                    # generation_config={"temperature": 0.7, "max_output_tokens": 4000} # Alternative way to set params
                ),
                timeout=AI_TIMEOUT
            )

            # Ensure response.text is used, and handle cases where response might not have it.
            content = response.text if hasattr(response, 'text') else ''
            if not content and response.parts: # Check parts if text is empty
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

    async def _fallback_to_available_ai(self, system_prompt: str, user_prompt: str, is_ipm: bool) -> Dict[str, str] | List[str]:
        logger.info(f"Attempting fallback for {'IPM' if is_ipm else 'Residia questions'}.")

        # Define preferred order
        preferred_order = [
            ("Claude", self.claude_client, self._generate_with_claude),
            ("OpenAI", self.openai_client, self._generate_with_openai),
            ("Gemini", self.gemini_model, self._generate_with_gemini),
        ]

        for name, client, method in preferred_order:
            if client:
                logger.info(f"Fallback: Trying {name}.")
                try:
                    # Ensure 'method' is awaited as they are async
                    return await method(system_prompt, user_prompt, is_ipm)
                except Exception as e:
                    logger.warning(f"Fallback to {name} failed: {type(e).__name__}: {e}")
            else:
                logger.debug(f"Fallback: {name} client not available.")

        logger.error("All AI fallbacks failed or no clients available.")
        # Return a mock response of the correct type if all fallbacks fail
        mock_type_str = "ipm" if is_ipm else "residia_questions"
        logger.warning(f"Returning emergency mock response for {mock_type_str} after all fallbacks failed.")
        return self._create_mock_response("EmergencyFallback", mock_type_str) # type: ignore

    def _validate_ipm_response(self, response: Optional[Dict[str, str]]) -> bool:
        if not response:
            logger.warning("Validation failed: Response is None.")
            return False

        required_keys = ["physical", "emotional", "unconscious", "counseling"]
        if not all(key in response for key in required_keys):
            logger.warning(f"Validation failed: Missing keys. Found: {list(response.keys())}, Expected: {required_keys}")
            return False

        filled_count = sum(1 for key in required_keys if response.get(key, "").strip())
        # As per issue: "at least 2 sections", and "Counseling section is mandatory"
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
        ai_model: Literal["claude", "openai", "gemini"] = "claude"
    ) -> Dict[str, str]:
        system_prompt, user_prompt = self._create_ipm_diagnosis_prompt(initial_prompt)
        max_retries = 2 # As per issue
        last_error: Optional[Exception] = None # Store the actual exception object

        for attempt in range(max_retries + 1):
            logger.info(f"Generating IPM diagnosis (AI: {ai_model}, Attempt: {attempt + 1}/{max_retries + 1})")
            result: Optional[Dict[str, str]] = None # Ensure result is Dict or None

            try:
                current_ai_method = None
                client_available = False

                if ai_model == "claude":
                    current_ai_method = self._generate_with_claude
                    client_available = bool(self.claude_client)
                elif ai_model == "openai":
                    current_ai_method = self._generate_with_openai
                    client_available = bool(self.openai_client)
                elif ai_model == "gemini":
                    current_ai_method = self._generate_with_gemini
                    client_available = bool(self.gemini_model)
                else:
                    logger.error(f"Invalid AI model specified: {ai_model}. Attempting general fallback.")
                    # Fall directly to general fallback if model name is wrong
                    result = await self._fallback_to_available_ai(system_prompt, user_prompt, is_ipm=True) # type: ignore

                if current_ai_method: # If a valid ai_model was specified
                    if client_available:
                        result = await current_ai_method(system_prompt, user_prompt, is_ipm=True) # type: ignore
                    else: # Client for specified AI not available
                        logger.warning(f"{ai_model} client not available for IPM diagnosis. Attempting specific fallback first.")
                        result = await self._fallback_to_available_ai(system_prompt, user_prompt, is_ipm=True) # type: ignore

                # Validate the result if one was obtained
                if self._validate_ipm_response(result):
                    logger.info("IPM diagnosis generated and validated successfully.")
                    return result # type: ignore # result is now confirmed Dict[str, str]
                else:
                    # Log warning if validation failed for a non-None result
                    if result is not None: # result could be {} or {"physical": "", ...}
                        logger.warning(f"IPM diagnosis response validation failed. Result: {str(result)[:300]}...")
                    # last_error should be an Exception or string if no exception occurred but validation failed
                    if not last_error: # Only set if not already set by an exception catcher
                        last_error = Exception("Invalid or empty response from AI after parsing and validation.")

            except Exception as e:
                logger.error(f"Error during IPM diagnosis generation (Attempt {attempt + 1}) with {ai_model}: {type(e).__name__}: {e}")
                last_error = e
                # If a specific API error or a config error (like ValueError from fallback),
                # and it's the last attempt, we might not want to just raise.
                # The original spec implies raising the last error.

            # Retry logic
            if attempt < max_retries:
                sleep_duration = 2 ** attempt
                logger.info(f"Retrying IPM diagnosis generation in {sleep_duration} seconds...")
                await asyncio.sleep(sleep_duration)
            elif attempt == max_retries and not self._validate_ipm_response(result): # Last attempt and still not valid
                 logger.warning(f"Final attempt for IPM diagnosis with {ai_model} failed validation. Trying general fallback if not already done.")
                 try:
                     # Ensure we are not in an infinite loop if _fallback_to_available_ai was already the source of 'result'
                     # This check is tricky. Assume _fallback_to_available_ai was tried if client_available was false.
                     if client_available : # Only try general fallback if the chosen AI was initially available but failed
                        result = await self._fallback_to_available_ai(system_prompt, user_prompt, is_ipm=True) # type: ignore
                        if self._validate_ipm_response(result):
                            logger.info("IPM diagnosis successfully generated via final fallback.")
                            return result # type: ignore
                        else:
                            if result is not None:
                                logger.warning(f"Final fallback for IPM diagnosis also failed validation. Result: {str(result)[:300]}...")
                            if not last_error: # If previous attempts didn't set an exception
                                last_error = Exception("Final fallback response for IPM was also invalid.")
                 except Exception as fallback_e:
                     logger.error(f"Exception during final fallback for IPM: {fallback_e}")
                     last_error = fallback_e


        logger.error(f"Failed to generate IPM diagnosis after all attempts and fallbacks. Last error: {last_error}")
        if isinstance(last_error, Exception):
            raise last_error
        else: # Should ideally always be an exception, but as a safeguard:
            raise Exception(f"IPM diagnosis generation failed with an unknown error or persistent validation failure: {last_error}")


    async def generate_residia_questions(
        self,
        session_data: Dict[str, any],
        ai_model: Literal["claude", "openai", "gemini"] = "claude"
    ) -> List[str]:
        system_prompt, user_prompt = self._create_residia_questions_prompt(session_data)
        max_retries = 1 # Per issue, less critical for complex retries
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            logger.info(f"Generating Residia questions (AI: {ai_model}, Attempt: {attempt + 1}/{max_retries + 1})")
            questions: Optional[List[str]] = None

            try:
                current_ai_method = None
                client_available = False

                if ai_model == "claude":
                    current_ai_method = self._generate_with_claude
                    client_available = bool(self.claude_client)
                elif ai_model == "openai":
                    current_ai_method = self._generate_with_openai
                    client_available = bool(self.openai_client)
                elif ai_model == "gemini":
                    current_ai_method = self._generate_with_gemini
                    client_available = bool(self.gemini_model)
                else:
                    logger.error(f"Invalid AI model for Residia questions: {ai_model}. Attempting fallback.")
                    questions = await self._fallback_to_available_ai(system_prompt, user_prompt, is_ipm=False) # type: ignore

                if current_ai_method:
                    if client_available:
                        questions = await current_ai_method(system_prompt, user_prompt, is_ipm=False) # type: ignore
                    else:
                        logger.warning(f"{ai_model} client not available for Residia questions. Attempting fallback.")
                        questions = await self._fallback_to_available_ai(system_prompt, user_prompt, is_ipm=False) # type: ignore

                # Validate questions
                if questions and len(questions) >= 3: # Expect at least 3 questions
                    logger.info(f"Residia questions generated successfully (count: {len(questions)}).")
                    return questions[:5] # Return up to 5
                else:
                    logger.warning(f"Residia questions generation resulted in too few questions or None. Count: {len(questions) if questions else 'None'}.")
                    if not last_error:
                        last_error = Exception("Insufficient or no questions generated by AI.")

            except Exception as e:
                logger.error(f"Error generating Residia questions (Attempt {attempt + 1}) with {ai_model}: {type(e).__name__}: {e}")
                last_error = e

            if attempt < max_retries:
                await asyncio.sleep(1) # Shorter sleep for question retries
            elif attempt == max_retries and not (questions and len(questions) >=3): # Final attempt failed
                logger.warning(f"Final attempt for Residia questions with {ai_model} failed. Trying general fallback if not already done.")
                try:
                    if client_available : # Only try general fallback if the chosen AI was initially available but failed
                        questions = await self._fallback_to_available_ai(system_prompt, user_prompt, is_ipm=False) # type: ignore
                        if questions and len(questions) >= 3:
                            logger.info(f"Residia questions successfully generated via final fallback (count: {len(questions)}).")
                            return questions[:5]
                        else:
                            logger.warning(f"Final fallback for Residia questions also failed to produce enough questions. Count: {len(questions) if questions else 'None'}")
                            if not last_error:
                                last_error = Exception("Final fallback for Residia questions also invalid.")
                except Exception as fallback_e:
                     logger.error(f"Exception during final fallback for Residia questions: {fallback_e}")
                     last_error = fallback_e


        logger.error(f"Failed to generate Residia questions after all attempts. Last error: {last_error}. Returning mock questions.")
        # Per issue, fallback to mock if all else fails for questions.
        return self._create_mock_response("CriticalFallbackMock", "residia_questions") # type: ignore

    # Placeholder for analyze_residia which is not part of this refactoring issue
    # This method was present in the original file, so keeping its structure.
    async def analyze_residia(
        self,
        session_data: Dict[str, any], # type: ignore
        user_answers: List[Dict[str, str]], # type: ignore
        identified_types: List[str], # type: ignore
        ai_model: Literal["claude", "openai", "gemini"] = "claude" # type: ignore
    ) -> str:
        logger.warning("analyze_residia method is a placeholder and not fully refactored in this task.")
        # This method would also use system_prompt, user_prompt, and call appropriate _generate_with_X
        # It should also have its own prompt creation, response parsing, and validation logic.
        # For now, returning a simple mock. This is outside the scope of the current issue.
        # from app.file_manager import get_file_content_for_ai # Assuming this import exists
        # residia_files = get_file_content_for_ai("residia") # Example: this would be part of its logic

        # Mocking a system and user prompt creation (very basic)
        _system_prompt = f"Analyze Residia for types: {', '.join(identified_types)} based on provided data."
        _user_prompt = f"Session: {session_data.get('initial_prompt', '')}. Answers: {user_answers}. Analyze."

        # Mock calling an AI - This is highly simplified
        if ai_model == "claude" and self.claude_client:
            # return await self._generate_with_claude(_system_prompt, _user_prompt, is_ipm=False) # This would need a new 'is_ipm' type
            pass # This would be a call to a specific residia analysis generation method
        # ... other AI models

        return f"Mock analysis for {ai_model} based on identified types: {', '.join(identified_types)}. User prompt: {session_data.get('initial_prompt', '')}. This function needs full implementation."

# Singleton instance:
# Depending on application structure (e.g., FastAPI), dependency injection might be used instead.
# For now, as per original structure, a direct instance can be created if needed elsewhere.
# However, it's often better to instantiate it where used or use a factory/DI.
# ai_service = AIService()
