import asyncio
import json
import os
import random
import re
from typing import Any, Dict, List, Optional

try:
    from groq import AsyncGroq
except Exception:  # pragma: no cover - optional dependency
    AsyncGroq = None


class LLMJudge:
    def __init__(self, groq_api_key: Optional[str] = None):
        self.max_retries = max(1, int(os.getenv("JUDGE_MAX_RETRIES", "3")))
        self.model_a_name = os.getenv("GROQ_JUDGE_MODEL_A", "llama-3.3-70b-versatile")
        self.model_b_name = os.getenv("GROQ_JUDGE_MODEL_B", "llama-3.1-8b-instant")

        key = (groq_api_key or os.getenv("GROQ_API_KEY") or "").strip()
        self.has_model_a = bool(key) and AsyncGroq is not None
        self.has_model_b = bool(key) and AsyncGroq is not None
        self.groq_client = AsyncGroq(api_key=key) if (key and AsyncGroq is not None) else None

        self.circuit_open = {
            "model_a": False,
            "model_b": False,
        }

    @staticmethod
    def _judge_system_prompt() -> str:
        return (
            "You are a strict benchmark judge for RAG answers. "
            "Score answer quality from 1 to 5. "
            "Use only this JSON schema: {\"score\": <number>, \"reasoning\": \"short reason\"}."
        )

    @staticmethod
    def _judge_user_prompt(question: str, answer: str, truth: str) -> str:
        return (
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Ground Truth:\n{truth}\n\n"
            "Return JSON only."
        )

    @staticmethod
    def _backoff_seconds(attempt: int, base: float) -> float:
        return base * (2 ** attempt) + random.uniform(0.0, 0.25)

    @staticmethod
    def _is_terminal_provider_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "resource_exhausted" in message
            or "quota" in message
            or "rate limit" in message
            or "429" in message
            or "api key" in message
            or "invalid" in message
            or "unauthorized" in message
            or "forbidden" in message
        )

    @staticmethod
    def _extract_numeric_score(text: str) -> float:
        if not text:
            return 3.0

        match = re.search(r"(\d+(\.\d+)?)", str(text))
        if not match:
            return 3.0

        score = float(match.group(1))
        return max(1.0, min(5.0, score))

    @staticmethod
    def _default_model_result(reasoning: str = "Fallback result") -> Dict[str, Any]:
        return {
            "score": 3.0,
            "reasoning": reasoning,
        }

    def _normalize_model_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return self._default_model_result()

        score = data.get("score", 3.0)
        try:
            score = float(score)
        except Exception:
            score = self._extract_numeric_score(str(score))

        score = max(1.0, min(5.0, score))
        reasoning = str(data.get("reasoning", "No reasoning"))

        return {
            "score": score,
            "reasoning": reasoning,
        }

    def _parse_json_payload(self, raw: str) -> Dict[str, Any]:
        try:
            if not raw:
                return self._default_model_result()

            text = raw.strip().replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                text = match.group(0)
            text = re.sub(r",\s*}", "}", text)

            data = json.loads(text)
            return self._normalize_model_result(data)
        except Exception:
            return self._default_model_result("Failed to parse judge output")

    async def call_groq_model(self, question: str, answer: str, truth: str, model_name: str) -> Dict[str, Any]:
        if self.groq_client is None:
            raise RuntimeError("Groq judge unavailable: missing GROQ_API_KEY or groq SDK")

        response = await self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": self._judge_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._judge_user_prompt(question, answer, truth),
                },
            ],
            temperature=0,
            max_tokens=220,
            response_format={"type": "json_object"},
        )

        raw = ""
        if response.choices:
            raw = response.choices[0].message.content or ""
        return self._parse_json_payload(raw)

    async def safe_model_score(self, question: str, answer: str, truth: str, label: str, model_name: str) -> Dict[str, Any]:
        if self.circuit_open.get(label):
            fallback = self._default_model_result("Circuit open")
            return {
                "score": float(fallback["score"]),
                "reasoning": str(fallback["reasoning"]),
                "is_live": False,
                "error": f"{label}_circuit_open",
            }

        if self.groq_client is None:
            missing_reason = "missing_groq_sdk" if AsyncGroq is None else "missing_groq_key"
            fallback = self._default_model_result("Groq unavailable")
            return {
                "score": float(fallback["score"]),
                "reasoning": str(fallback["reasoning"]),
                "is_live": False,
                "error": missing_reason,
            }

        last_error: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                result = await self.call_groq_model(question, answer, truth, model_name)
                return {
                    "score": float(result.get("score", 3.0)),
                    "reasoning": str(result.get("reasoning", "")),
                    "is_live": True,
                    "error": None,
                }
            except Exception as exc:
                last_error = str(exc)[:300]
                print(f"{label} attempt {attempt + 1} failed: {exc}")
                if self._is_terminal_provider_error(exc):
                    self.circuit_open[label] = True
                    break
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self._backoff_seconds(attempt, base=0.9))

        fallback = self._default_model_result("Retry exhausted")
        return {
            "score": float(fallback["score"]),
            "reasoning": str(fallback["reasoning"]),
            "is_live": False,
            "error": last_error or f"{label}_failed",
        }

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        model_a = await self.safe_model_score(
            question=question,
            answer=answer,
            truth=ground_truth,
            label="model_a",
            model_name=self.model_a_name,
        )
        model_b = await self.safe_model_score(
            question=question,
            answer=answer,
            truth=ground_truth,
            label="model_b",
            model_name=self.model_b_name,
        )

        score_a = float(model_a.get("score", 3.0))
        score_b = float(model_b.get("score", 3.0))
        diff = abs(score_a - score_b)

        agreement_rate = 1.0 if diff <= 1 else max(0.0, 1 - (diff - 1) / 4)
        final_score = (score_a + score_b) / 2

        live_provider_count = int(bool(model_a.get("is_live"))) + int(bool(model_b.get("is_live")))
        provider_errors = {
            self.model_a_name: model_a.get("error"),
            self.model_b_name: model_b.get("error"),
        }

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": round(agreement_rate, 2),
            "individual_scores": {
                self.model_a_name: round(score_a, 2),
                self.model_b_name: round(score_b, 2),
            },
            "reasoning": {
                self.model_a_name: str(model_a.get("reasoning", "")),
                self.model_b_name: str(model_b.get("reasoning", "")),
            },
            "provider_status": {
                self.model_a_name: "live" if model_a.get("is_live") else "fallback",
                self.model_b_name: "live" if model_b.get("is_live") else "fallback",
            },
            "provider_errors": {k: v for k, v in provider_errors.items() if v},
            "live_provider_count": live_provider_count,
        }

    async def evaluate_batch(self, dataset: List[Dict[str, Any]], batch_size: int = 2) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(batch_size)

        async def worker(case: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.evaluate_multi_judge(
                    case.get("question", ""),
                    case.get("answer", ""),
                    case.get("expected_answer", ""),
                )

        tasks = [worker(case) for case in dataset]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":

    async def test() -> None:
        judge = LLMJudge()
        result = await judge.evaluate_multi_judge(
            "Thu do Viet Nam la gi?",
            "Thu do Viet Nam la Ha Noi.",
            "Ha Noi",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(test())
