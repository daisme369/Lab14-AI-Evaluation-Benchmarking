import asyncio
import importlib
import os
import json
import random
import re
import statistics
import re
from typing import Dict, Any, List

import google.genai as genai
from google.genai import types
from groq import AsyncGroq
from dotenv import load_dotenv

openai = importlib.import_module("openai")


class LLMJudge:
    def __init__(self, gemini_api_key=None, groq_api_key=None):
        self.gemini_client = genai.Client(
            api_key=gemini_api_key or os.getenv("GEMINI_API_KEY")
        )

        self.groq_client = AsyncGroq(
            api_key=groq_api_key or os.getenv("GROQ_API_KEY")
        )

    # =====================================================
    # PROMPTS
    # =====================================================

    def gemini_prompt(self, question, answer, truth):
        return f"""
Question:
{question}

Answer:
{answer}

Ground Truth:
{truth}

Return ONLY one number score from 1 to 5.
"""

    def groq_system_prompt(self):
        return """
Evaluate answer quality.

Return ONLY valid JSON:

{
  "overall_score": 4.2,
  "criteria_scores": {
    "accuracy": 4,
    "professionalism": 5,
    "safety": 5
  },
  "reasoning": "short reason"
}
"""

    def groq_user_prompt(self, question, answer, truth):
        return f"""
Question:
{question}

Answer:
{answer}

Ground Truth:
{truth}
"""

    # =====================================================
    # HELPERS
    # =====================================================

    def parse_number(self, text):

        if not text:
            return 3.0

        match = re.search(r"(\d+(\.\d+)?)", str(text))

        if not match:
            return 3.0

        score = float(match.group(1))

        return max(1.0, min(5.0, score))

    def normalize_groq_result(self, data):

        if not isinstance(data, dict):
            return self.default_groq()

        if "overall_score" not in data:
            data["overall_score"] = 3.0

        if "reasoning" not in data:
            data["reasoning"] = "No reasoning"

        if "criteria_scores" not in data:
            data["criteria_scores"] = {
                "accuracy": 3,
                "professionalism": 3,
                "safety": 3
            }

        return data

    def default_groq(self):
        return {
            "overall_score": 3.0,
            "criteria_scores": {
                "accuracy": 3,
                "professionalism": 3,
                "safety": 3
            },
            "reasoning": "Fallback result"
        }

    def parse_json(self, raw):

        try:
            if not raw:
                return self.default_groq()

            raw = raw.strip()
            raw = raw.replace("```json", "")
            raw = raw.replace("```", "")
            raw = raw.strip()

            match = re.search(r"\{[\s\S]*\}", raw)

            if match:
                raw = match.group(0)

            raw = re.sub(r",\s*}", "}", raw)

            data = json.loads(raw)

            return self.normalize_groq_result(data)

        except Exception:
            return self.default_groq()

    # =====================================================
    # GEMINI
    # =====================================================

    async def call_gemini_score(self, question, answer, truth):

        prompt = self.gemini_prompt(question, answer, truth)

        def sync_call():
            return self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=20
                ),
            )

        response = await asyncio.to_thread(sync_call)

        text = None

        if getattr(response, "text", None):
            text = response.text

        if text is None:
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception:
                pass

        return self.parse_number(text)

    # =====================================================
    # GROQ
    # =====================================================

    async def call_groq_judge(self, question, answer, truth):

        response = await self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": self.groq_system_prompt()
                },
                {
                    "role": "user",
                    "content": self.groq_user_prompt(
                        question,
                        answer,
                        truth
                    )
                }
            ],
            temperature=0,
            max_tokens=250,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content

        return self.parse_json(raw)

    # =====================================================
    # SAFE WRAPPERS
    # =====================================================

    async def safe_gemini(self, question, answer, truth):

        for attempt in range(3):
            try:
                return await self.call_gemini_score(
                    question,
                    answer,
                    truth
                )
            except Exception as e:
                print(f"gemini attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)

        return 3.0

    async def safe_groq(self, question, answer, truth):

        for attempt in range(3):
            try:
                return await self.call_groq_judge(
                    question,
                    answer,
                    truth
                )
            except Exception as e:
                print(f"groq attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2)

        return self.default_groq()

    # =====================================================
    # MAIN
    # =====================================================

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str
    ):

        gemini_score = await self.safe_gemini(
            question,
            answer,
            ground_truth
        )

        groq_res = await self.safe_groq(
            question,
            answer,
            ground_truth
        )

        groq_score = float(groq_res.get("overall_score", 3.0))

        diff = abs(gemini_score - groq_score)

        agreement_rate = 1.0 if diff <= 1 else max(
            0.0,
            1 - (diff - 1) / 4
        )

        final_score = (gemini_score + groq_score) / 2

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": round(agreement_rate, 2),
            "individual_scores": {
                "gemini": round(gemini_score, 2),
                "groq": round(groq_score, 2),
            },
            "reasoning": {
                "groq": groq_res.get("reasoning", "")
            }
        }

    # =====================================================
    # BATCH
    # =====================================================

    async def evaluate_batch(
        self,
        dataset: List[Dict],
        batch_size: int = 2
    ):

        semaphore = asyncio.Semaphore(batch_size)

        async def worker(case):
            async with semaphore:
                return await self.evaluate_multi_judge(
                    case.get("question", ""),
                    case.get("answer", ""),
                    case.get("expected_answer", ""),
                )

        tasks = [worker(case) for case in dataset]

        return await asyncio.gather(*tasks)


if __name__ == "__main__":

    async def test():

        judge = LLMJudge()

        res = await judge.evaluate_multi_judge(
            "Thủ đô Việt Nam là gì?",
            "Thủ đô Việt Nam là Hà Nội.",
            "Hà Nội"
        )

        print(json.dumps(res, indent=2, ensure_ascii=False))

    asyncio.run(test())