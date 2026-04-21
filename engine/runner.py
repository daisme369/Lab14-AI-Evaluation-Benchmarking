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
            "agent_response": answer,
            "latency": round(latency, 3),
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
            "agent_response": "",
            "latency": 0,
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
        batch_size: int = 1
    ) -> List[Dict]:
        """
        Stable mode:
        - default batch_size = 1 tránh Groq RPM limit
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
