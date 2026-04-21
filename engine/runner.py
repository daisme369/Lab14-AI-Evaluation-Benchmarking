import asyncio
import time
from typing import Any, Dict, List


class BenchmarkRunner:
    def __init__(self, agent: Any, evaluator: Any, judge: Any, pass_score_threshold: float = 3.0):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.pass_score_threshold = pass_score_threshold

    def _build_error_result(self, test_case: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        return {
            "test_case": test_case.get("question", ""),
            "case_id": (test_case.get("metadata") or {}).get("case_id"),
            "metadata": test_case.get("metadata", {}),
            "agent_response": "",
            "retrieved_ids": [],
            "latency": 0.0,
            "ragas": {
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "retrieval": {"hit_rate": 0.0, "mrr": 0.0},
                "error": error_message,
            },
            "judge": {
                "final_score": 0.0,
                "agreement_rate": 0.0,
                "individual_scores": {},
                "reasoning": {"error": error_message},
            },
            "status": "error",
            "error": error_message,
        }

    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")

        try:
            response = await self.agent.query(question)
        except Exception as exc:
            return self._build_error_result(test_case, f"agent_error: {exc}")

        latency = time.perf_counter() - start_time
        if not isinstance(response, dict):
            response = {"answer": str(response), "retrieved_ids": []}

        answer = response.get("answer", "")
        retrieved_ids = response.get("retrieved_ids", [])

        try:
            ragas_scores = await self.evaluator.score(test_case, response)
        except Exception as exc:
            ragas_scores = {
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "retrieval": {"hit_rate": 0.0, "mrr": 0.0},
                "error": f"evaluator_error: {exc}",
            }

        try:
            judge_result = await self.judge.evaluate_multi_judge(question, answer, expected_answer)
        except Exception as exc:
            judge_result = {
                "final_score": 0.0,
                "agreement_rate": 0.0,
                "individual_scores": {},
                "reasoning": {"error": f"judge_error: {exc}"},
            }

        final_score = float(judge_result.get("final_score", 0.0))
        retrieval_scores = ragas_scores.get("retrieval", {})
        hit_rate = float(retrieval_scores.get("hit_rate", 0.0))
        status = "pass" if final_score >= self.pass_score_threshold and hit_rate > 0.0 else "fail"

        return {
            "test_case": question,
            "case_id": (test_case.get("metadata") or {}).get("case_id"),
            "metadata": test_case.get("metadata", {}),
            "agent_response": answer,
            "retrieved_ids": retrieved_ids,
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": status,
        }

    async def run_all(self, dataset: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Run the benchmark in async batches to control request burst and rate limits.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        results: List[Dict[str, Any]] = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tasks = [asyncio.create_task(self.run_single_test(case)) for case in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for index, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append(self._build_error_result(batch[index], f"runner_error: {result}"))
                    continue
                results.append(result)

        return results
