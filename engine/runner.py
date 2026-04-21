import asyncio
import time
from typing import List, Dict, Any


class BenchmarkRunner:
    def __init__(self, agent: Any, evaluator: Any, judge: Any, pass_score_threshold: float = 3.0):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.pass_score_threshold = pass_score_threshold

    # =====================================================
    # RUN SINGLE TEST
    # =====================================================

    async def run_single_test(self, test_case: Dict) -> Dict:

        start_time = time.perf_counter()

        # 1. Call Agent
        response = await self.agent.query(test_case["question"])

        latency = time.perf_counter() - start_time

        # normalize response
        if isinstance(response, dict):
            answer = response.get("answer", "")
        else:
            answer = str(response)
            response = {"answer": answer}

        retrieved_ids = response.get("retrieved_ids", []) if isinstance(response, dict) else []
        case_metadata = test_case.get("metadata", {})
        response_metadata = response.get("metadata", {}) if isinstance(response, dict) else {}
        agent_tokens = int(response_metadata.get("tokens_used", 0) or 0)

        # 2. Retrieval / RAGAS
        ragas_scores = await self.evaluator.score(
            test_case,
            response
        )

        # 3. Multi Judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            answer,
            test_case["expected_answer"]
        )

        final_score = float(judge_result.get("final_score", 3.0))

        return {
            "test_case": test_case["question"],
            "case_id": case_metadata.get("case_id"),
            "metadata": case_metadata,
            "agent_response": answer,
            "retrieved_ids": retrieved_ids,
            "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
            "latency": round(latency, 3),
            "token_usage": {
                "agent_tokens": agent_tokens,
            },
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if final_score < 3 else "pass"
        }

    # =====================================================
    # SAFE WRAPPER
    # =====================================================

    async def safe_run_single_test(
        self,
        test_case: Dict,
        idx: int,
        total: int
    ) -> Dict:

        last_error = None

        for attempt in range(2):
            try:
                print(f"🔹 Running test {idx}/{total}")

                result = await self.run_single_test(test_case)

                print(f"✅ Done test {idx}/{total}")

                return result

            except Exception as e:
                last_error = e
                print(
                    f"❌ Test {idx}/{total} failed "
                    f"(attempt {attempt+1}): {e}"
                )
                await asyncio.sleep(3)

        return {
            "test_case": test_case.get("question", ""),
            "case_id": (test_case.get("metadata") or {}).get("case_id"),
            "metadata": test_case.get("metadata", {}),
            "agent_response": "",
            "retrieved_ids": [],
            "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
            "latency": 0,
            "token_usage": {
                "agent_tokens": 0,
            },
            "ragas": {},
            "judge": {
                "final_score": 0,
                "agreement_rate": 0
            },
            "status": "error",
            "error": str(last_error)
        }

    # =====================================================
    # RUN ALL
    # =====================================================

    async def run_all(
        self,
        dataset: List[Dict],
        batch_size: int = 5
    ) -> List[Dict]:
        """
        Stable mode:
        - default batch_size = 5
        - sleep giữa batch
        """

        results = []
        total = len(dataset)

        for i in range(0, total, batch_size):

            batch = dataset[i:i + batch_size]

            tasks = [
                self.safe_run_single_test(
                    case,
                    i + j + 1,
                    total
                )
                for j, case in enumerate(batch)
            ]

            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

            # tránh rate limit
            await asyncio.sleep(2)

        print("🎉 Benchmark completed.")

        return results
