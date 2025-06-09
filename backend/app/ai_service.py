import os
import re
import asyncio
import logging
from typing import Dict, List, Optional, Literal, Tuple, Union
from dotenv import load_dotenv

import anthropic
import openai
from google import generativeai as genai
from google.api_core import exceptions as google_exceptions

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", 300))

class InsufficientPlanError(Exception):
    """プランが不十分な場合のエラー"""
    pass

def validate_plan_for_ai(plan_type: str, requested_ai: str) -> bool:
    if plan_type == "basic" and requested_ai == "claude": return False
    return True

class ResponseParser:
    @staticmethod
    def parse_ipm_diagnosis(content: str) -> Dict[str, str]:
        result = {"physical": "", "emotional": "", "unconscious": "", "counseling": ""}
        patterns = {
            'physical': r'###PHYSICAL_START###\s*(.*?)\s*###PHYSICAL_END###',
            'emotional': r'###EMOTIONAL_START###\s*(.*?)\s*###EMOTIONAL_END###',
            'unconscious': r'###UNCONSCIOUS_START###\s*(.*?)\s*###UNCONSCIOUS_END###',
            'counseling': r'###COUNSELING_START###\s*(.*?)\s*###COUNSELING_END###'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match: result[key] = match.group(1).strip()
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
                if "###" not in extracted_text: return extracted_text
        logger.debug(f"Fallback extraction failed for section: {section_type}")
        return ""

    @staticmethod
    def parse_residia_questions(content: str) -> List[str]:
        questions = []
        for i in range(1, 6):
            pattern = rf'###Q{i}_START###\s*(.*?)\s*###Q{i}_END###'
            match = re.search(pattern, content, re.DOTALL)
            if match: questions.append(match.group(1).strip())
        if len(questions) < 3:
            logger.warning("Marker-based extraction for Residia questions yielded less than 3 questions. Attempting fallback.")
            fallback_pattern = r'(?:^|\n)\s*(?:(?:\d+\.?|\*|-)\s+)?(.*?)(?=\n\s*(?:(?:\d+\.?|\*|-)\s+)|$|###Q\d_START###)'
            fallback_matches = re.findall(fallback_pattern, content, re.MULTILINE)
            extracted_fallback_questions = [
                match.strip() for match_text in fallback_matches
                if (cleaned_q := match_text.strip()) and len(cleaned_q) > 10 and not cleaned_q.startswith("###") and not cleaned_q.endswith("###")
            ]
            if questions:
                for fq in extracted_fallback_questions:
                    if fq not in questions and len(questions) < 5: questions.append(fq)
            else: questions = extracted_fallback_questions
        return questions[:5]

class AIService:
    def __init__(self):
        self.claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY and "dummy" not in ANTHROPIC_API_KEY else None
        if not self.claude_client: logger.warning("Anthropic API key not available or is a dummy key.")
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and "dummy" not in OPENAI_API_KEY else None
        if not self.openai_client: logger.warning("OpenAI API key not available or is a dummy key.")
        self.gemini_model = None
        if GOOGLE_API_KEY and "dummy" not in GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e: logger.error(f"Failed to configure Gemini: {e}")
        if not self.gemini_model: logger.warning("Google API key not available or is a dummy key.")

    def get_ai_models_for_plan(self, plan_type: str) -> tuple[str, list[str]]: # For IPM
        from app.config import AI_MODEL_BY_PLAN, DEFAULT_FALLBACK_ORDER
        config = AI_MODEL_BY_PLAN.get(plan_type)
        if config and config.get("primary") and config.get("fallback_order"):
            return config["primary"], config["fallback_order"]
        logger.warning(f"Plan '{plan_type}' not found or malformed in AI_MODEL_BY_PLAN for IPM. Using default.");
        return DEFAULT_FALLBACK_ORDER[0], DEFAULT_FALLBACK_ORDER

    def _get_ai_model_for_plan(self, plan_type: str) -> str: # For Residia (new spec)
        model_mapping = {"basic": "gemini", "advance": "openai", "maestro": "claude"}
        return model_mapping.get(plan_type, "claude")

    def _get_fallback_order_for_plan(self, plan_type: str) -> List[str]: # For Residia (new spec)
        fallback_orders = {
            "basic": ["gemini", "openai", "claude"],
            "advance": ["openai", "gemini", "claude"],
            "maestro": ["claude", "openai", "gemini"]
        }
        return fallback_orders.get(plan_type, ["claude", "openai", "gemini"])

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

    def _create_residia_analysis_prompt(self, session_data: dict, user_answers: List[str], identified_types: List[str]) -> tuple[str, str]:
        system_prompt = """あなたは幼少期トラウマ（レジディア）分析の専門家です。
ユーザーの回答から、特定されたレジディアタイプについて詳細な分析を提供します。
分析は3000-6000字で、以下の構成で記述してください：
1. 特定されたレジディアタイプの説明（各500-800字）
2. 現在の症状との関連性（800-1200字）
3. 治癒への道筋と具体的なワーク（1000-1500字）
4. 統合的なメッセージ（700-1000字）"""
        user_prompt = f"""以下の情報を基にレジディア分析を行ってください：
【初期症状】
{session_data.get('initial_prompt', '')}
【IPM診断結果】
肉体的診断: {session_data.get('physical_diagnosis', '')}
感情的診断: {session_data.get('emotional_diagnosis', '')}
無意識の診断: {session_data.get('unconscious_diagnosis', '')}
【レジディア質問への回答】
{chr(10).join([f"Q{i+1}: {ans}" for i, ans in enumerate(user_answers)])}
【特定されたレジディアタイプ】
{', '.join(identified_types)}
必ず3000-6000字で、上記の構成に従って分析を記述してください。"""
        return system_prompt, user_prompt

    def _create_mock_response(self, ai_type: str, type: str = "ipm") -> Dict[str, str] | List[str]:
        logger.info(f"Creating general mock response for {ai_type} ({type})")
        if type == "ipm": return {"physical": f"Mock physical for {ai_type}", "emotional": f"Mock emotional for {ai_type}", "unconscious": f"Mock unconscious for {ai_type}", "counseling": f"Mock counseling for {ai_type}"}
        elif type == "residia_questions": return [f"Mock Q1 for {ai_type}", f"Mock Q2 for {ai_type}"]
        return {}

    def _create_mock_residia_questions(self) -> List[str]:
        logger.info("Creating mock Residia questions.")
        return ["Mock Q1 residia", "Mock Q2 residia", "Mock Q3 residia", "Mock Q4 residia", "Mock Q5 residia"]

    def _create_mock_residia_analysis(self, identified_types: List[str]) -> str:
        logger.info(f"Creating mock Residia analysis for types: {identified_types}")
        analysis = f"""【レジディア分析レポート】 特定されたレジディアタイプ：{', '.join(identified_types) if identified_types else "タイプ特定なし"} ... (detailed analysis) ..."""
        base_len = len(analysis)
        repetitions = (3000 // base_len) + 1 if base_len > 0 else 200
        final_analysis = (analysis + "\n\n") * repetitions
        return final_analysis[:6000]

    async def _generate_with_claude(self, system_prompt: str, user_prompt: str) -> str:
        if not self.claude_client: raise ConnectionError("Claude client not available.")
        logger.info("Calling Claude API for raw content")
        response = await asyncio.wait_for(asyncio.to_thread(self.claude_client.messages.create, model="claude-3-opus-20240229", max_tokens=4000, temperature=0.7, system=system_prompt, messages=[{"role": "user", "content": user_prompt}]), timeout=AI_TIMEOUT)
        content = response.content[0].text
        logger.debug(f"Claude raw response (first 1000 chars): {content[:1000]}")
        return content

    async def _generate_with_openai(self, system_prompt: str, user_prompt: str) -> str:
        if not self.openai_client: raise ConnectionError("OpenAI client not available.")
        logger.info("Calling OpenAI API for raw content")
        response = await asyncio.wait_for(asyncio.to_thread(self.openai_client.chat.completions.create, model="gpt-4", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.7, max_tokens=4000), timeout=AI_TIMEOUT)
        content = response.choices[0].message.content
        if content is None: raise ValueError("OpenAI API returned None content.")
        logger.debug(f"OpenAI raw response (first 1000 chars): {content[:1000]}")
        return content

    async def _generate_with_gemini(self, system_prompt: str, user_prompt: str) -> str:
        if not self.gemini_model: raise ConnectionError("Gemini model not available.")
        logger.info("Calling Gemini API for raw content")
        effective_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = await asyncio.wait_for(asyncio.to_thread(self.gemini_model.generate_content, effective_prompt), timeout=AI_TIMEOUT)
        content = response.text if hasattr(response, 'text') else ''
        if not content and hasattr(response, 'parts') and response.parts: content = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        if not content: raise ValueError("Gemini API returned empty content.")
        logger.debug(f"Gemini raw response (first 1000 chars): {content[:1000]}")
        return content

    def _legacy_parse_claude_response(self, content: str) -> Dict[str, str]:
        logger.info("Using legacy Claude parser for IPM diagnosis.")
        result = {"physical": "", "emotional": "", "unconscious": "", "counseling": ""}
        physical_match = re.search(r"(?:肉体的な要因|肉体的要因|1\.)[:：\s]*(.*?)(?=\n\n(?:感情的な要因|感情的要因|2\.)|$)", content, re.DOTALL | re.IGNORECASE);
        if physical_match: result['physical'] = physical_match.group(1).strip()
        emotional_match = re.search(r"(?:感情的な要因|感情的要因|2\.)[:：\s]*(.*?)(?=\n\n(?:無意識の要因|無意識的要因|3\.)|$)", content, re.DOTALL | re.IGNORECASE);
        if emotional_match: result['emotional'] = emotional_match.group(1).strip()
        unconscious_match = re.search(r"(?:無意識の要因|無意識的要因|3\.)[:：\s]*(.*?)(?=\n\n(?:総合的なカウンセリング|カウンセリング|4\.)|$)", content, re.DOTALL | re.IGNORECASE);
        if unconscious_match: result['unconscious'] = unconscious_match.group(1).strip()
        counseling_match = re.search(r"(?:総合的なカウンセリング|カウンセリング|4\.)[:：\s]*(.*)", content, re.DOTALL | re.IGNORECASE);
        if counseling_match: result['counseling'] = counseling_match.group(1).strip()
        return result

    def _validate_ipm_response(self, response: Optional[Dict[str, str]]) -> bool:
        if not response: logger.warning("Validation failed: Response is None or empty."); return False
        required_keys = ["physical", "emotional", "unconscious", "counseling"]
        if not all(key in response for key in required_keys): logger.warning(f"Validation failed: Missing keys."); return False
        filled_count = sum(1 for key in required_keys if response.get(key, "").strip())
        if filled_count < 2: logger.warning(f"Validation failed: Insufficient content."); return False
        if not response.get("counseling", "").strip(): logger.warning("Validation failed: Counseling section empty."); return False
        logger.info("IPM response validation successful."); return True

    async def generate_ipm_diagnosis( self, initial_prompt: str, ai_model: Optional[str] = None, plan_type: Optional[str] = None ) -> Dict[str, str]:
        from app.config import DEFAULT_FALLBACK_ORDER
        primary_model_to_use: Optional[str] = None; current_fallback_order: List[str] = []
        if ai_model:
            primary_model_to_use = ai_model
            current_fallback_order = [ai_model] + [m for m in DEFAULT_FALLBACK_ORDER if m != ai_model]
        elif plan_type: _primary, _fallback_order = self.get_ai_models_for_plan(plan_type); primary_model_to_use = _primary; current_fallback_order = _fallback_order
        else: raise ValueError("Either ai_model or plan_type must be specified for IPM diagnosis")
        logger.info(f"IPM Diagnosis - Plan: {plan_type}, Explicit: {ai_model}, Primary: {primary_model_to_use}, Order: {current_fallback_order}")
        system_prompt, user_prompt = self._create_ipm_diagnosis_prompt(initial_prompt)
        last_error: Optional[Exception] = None
        for model_name_in_order in current_fallback_order:
            try:
                logger.info(f"Attempting IPM diagnosis with {model_name_in_order}")
                content = ""
                if model_name_in_order == "claude": content = await self._generate_with_claude(system_prompt, user_prompt)
                elif model_name_in_order == "openai": content = await self._generate_with_openai(system_prompt, user_prompt)
                elif model_name_in_order == "gemini": content = await self._generate_with_gemini(system_prompt, user_prompt)
                else: logger.warning(f"AI client for {model_name_in_order} not available or model unknown. Skipping."); continue

                result: Dict[str, str]
                if "###PHYSICAL_START###" in content:
                    result = ResponseParser.parse_ipm_diagnosis(content)
                elif model_name_in_order == "claude":
                     result = self._legacy_parse_claude_response(content)
                else:
                    result = ResponseParser.parse_ipm_diagnosis(content)

                if self._validate_ipm_response(result): logger.info(f"IPM diagnosis successful with {model_name_in_order}."); return result
                else: last_error = Exception(f"Response validation failed for {model_name_in_order}.")
            except ConnectionError as e: logger.warning(f"{model_name_in_order} client not available: {e}"); last_error = e; continue
            except Exception as e: logger.error(f"Error with {model_name_in_order} for IPM: {e}"); last_error = e

        logger.error(f"All AI models for IPM diagnosis failed. Last error: {last_error}. Fallback Order: {current_fallback_order}")
        if isinstance(last_error, Exception): raise last_error
        raise Exception("IPM diagnosis failed after trying all fallbacks.")

    async def generate_residia_questions(
        self,
        session_data: dict,
        ai_model: Optional[str] = None,
        plan_type: Optional[str] = None
    ) -> List[str]:
        from app.config import DEFAULT_FALLBACK_ORDER

        chosen_ai_model: str
        current_fallback_order: List[str]

        if ai_model:
            chosen_ai_model = ai_model
            if chosen_ai_model in DEFAULT_FALLBACK_ORDER:
                temp_order = [m for m in DEFAULT_FALLBACK_ORDER if m != chosen_ai_model]
                current_fallback_order = [chosen_ai_model] + temp_order
            else:
                current_fallback_order = [chosen_ai_model] + DEFAULT_FALLBACK_ORDER
            logger.info(f"Residia Questions: Explicit AI model '{chosen_ai_model}' specified. Fallback order: {current_fallback_order}")
        elif plan_type:
            chosen_ai_model = self._get_ai_model_for_plan(plan_type)
            current_fallback_order = self._get_fallback_order_for_plan(plan_type)
            logger.info(f"Residia Questions: Plan-based AI for plan '{plan_type}'. Primary: {chosen_ai_model}, Order: {current_fallback_order}")
        else:
            logger.warning("Residia Questions: Neither ai_model nor plan_type specified. Using default Claude.")
            chosen_ai_model = "claude"
            current_fallback_order = ["claude", "openai", "gemini"]

        system_prompt, user_prompt = self._create_residia_questions_prompt(session_data)
        last_error: Optional[Exception] = None

        for attempt, model_name_in_order in enumerate(current_fallback_order):
            try:
                logger.info(f"Attempting Residia questions with {model_name_in_order} (Plan: {plan_type or 'N/A'}, Attempt: {attempt + 1}/{len(current_fallback_order)})")
                content = ""
                if model_name_in_order == "claude": content = await self._generate_with_claude(system_prompt, user_prompt)
                elif model_name_in_order == "openai": content = await self._generate_with_openai(system_prompt, user_prompt)
                elif model_name_in_order == "gemini": content = await self._generate_with_gemini(system_prompt, user_prompt)
                else:
                    logger.warning(f"Client for {model_name_in_order} (Residia questions) not available or model unknown. Trying next.")
                    if not last_error: last_error = Exception(f"Client for {model_name_in_order} not available.")
                    continue

                questions = ResponseParser.parse_residia_questions(content)

                if len(questions) >= 3:
                    logger.info(f"Residia questions successful with {model_name_in_order} ({len(questions)} questions).")
                    return questions[:5]
                else:
                    logger.warning(f"Insufficient questions from {model_name_in_order} for Residia. Count: {len(questions) if questions else 'None'}.")
                    last_error = Exception(f"Insufficient questions from {model_name_in_order}.")

            except ConnectionError as e:
                logger.warning(f"{model_name_in_order} client not available for Residia Questions: {e}");
                last_error = e;
                if attempt < len(current_fallback_order) -1 : await asyncio.sleep(1);
                continue
            except Exception as e:
                logger.error(f"Error generating Residia questions with {model_name_in_order}: {type(e).__name__} - {e}")
                last_error = e
                if attempt < len(current_fallback_order) - 1:
                    await asyncio.sleep(2 ** attempt)

        logger.error(f"All AI models in fallback order {current_fallback_order} failed for Residia questions. Last error: {last_error}. Returning mock questions.")
        return self._create_mock_residia_questions()

    async def analyze_residia(
        self,
        session_data: dict,
        user_answers: List[str],
        identified_types: List[str],
        ai_model: Optional[str] = None,
        plan_type: Optional[str] = None
    ) -> str:
        """レジディア分析を実行（3000-6000字）"""
        from app.config import DEFAULT_FALLBACK_ORDER

        chosen_ai_model: str
        current_fallback_order: List[str]

        if ai_model:
            chosen_ai_model = ai_model
            if chosen_ai_model in DEFAULT_FALLBACK_ORDER:
                temp_order = [m for m in DEFAULT_FALLBACK_ORDER if m != chosen_ai_model]
                current_fallback_order = [chosen_ai_model] + temp_order
            else:
                current_fallback_order = [chosen_ai_model] + DEFAULT_FALLBACK_ORDER
            logger.info(f"Residia Analysis: Explicit AI model '{chosen_ai_model}' specified. Fallback order: {current_fallback_order}")
        elif plan_type:
            chosen_ai_model = self._get_ai_model_for_plan(plan_type)
            current_fallback_order = self._get_fallback_order_for_plan(plan_type)
            logger.info(f"Residia Analysis: Plan-based AI for plan '{plan_type}'. Primary: {chosen_ai_model}, Order: {current_fallback_order}")
        else:
            logger.warning("Residia Analysis: Neither ai_model nor plan_type specified. Using default Claude.")
            chosen_ai_model = "claude"
            current_fallback_order = ["claude", "openai", "gemini"]

        system_prompt, user_prompt = self._create_residia_analysis_prompt(session_data, user_answers, identified_types)
        last_error: Optional[Exception] = None
        analysis_result: str = ""

        for attempt, model_name_in_order in enumerate(current_fallback_order):
            try:
                logger.info(f"Attempting Residia analysis with {model_name_in_order} (Plan: {plan_type or 'N/A'}, Attempt: {attempt + 1}/{len(current_fallback_order)})")
                content = ""
                if model_name_in_order == "claude": content = await self._generate_with_claude(system_prompt, user_prompt)
                elif model_name_in_order == "openai": content = await self._generate_with_openai(system_prompt, user_prompt)
                elif model_name_in_order == "gemini": content = await self._generate_with_gemini(system_prompt, user_prompt)
                else:
                    logger.warning(f"Client for {model_name_in_order} (Residia analysis) not available or model unknown. Trying next.")
                    if not last_error: last_error = Exception(f"Client for {model_name_in_order} not available.")
                    continue

                analysis_result = content
                if 3000 <= len(analysis_result) <= 6000:
                    logger.info(f"Residia analysis successful with {model_name_in_order} ({len(analysis_result)} chars).")
                    return analysis_result
                elif len(analysis_result) > 6000:
                    logger.warning(f"Residia analysis from {model_name_in_order} too long ({len(analysis_result)} chars). Truncating.")
                    return analysis_result[:6000]
                else:
                    logger.warning(f"Residia analysis from {model_name_in_order} too short ({len(analysis_result)} chars). Trying next model.")
                    last_error = Exception(f"Analysis too short from {model_name_in_order} ({len(analysis_result)} chars).")

            except ConnectionError as e:
                logger.warning(f"{model_name_in_order} client not available for Residia Analysis: {e}")
                last_error = e
                if attempt < len(current_fallback_order) -1 : await asyncio.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Error generating Residia analysis with {model_name_in_order}: {type(e).__name__} - {e}")
                last_error = e
                if attempt < len(current_fallback_order) - 1:
                    await asyncio.sleep(2 ** attempt)

        logger.error(f"All AI models in fallback order {current_fallback_order} failed to provide a valid Residia analysis. Last error: {last_error}.")
        return self._create_mock_residia_analysis(identified_types)

ai_service = AIService()
