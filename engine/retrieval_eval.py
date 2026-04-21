from typing import Dict, List
import asyncio
import re

class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    @staticmethod
    def _lexical_overlap(answer: str, expected_answer: str) -> float:
        answer_tokens = set(re.findall(r"[a-z0-9_]+", (answer or "").lower()))
        expected_tokens = set(re.findall(r"[a-z0-9_]+", (expected_answer or "").lower()))
        if not expected_tokens:
            return 0.0
        return len(answer_tokens.intersection(expected_tokens)) / len(expected_tokens)

    @staticmethod
    def _extract_retrieved_ids(response: Dict) -> List[str]:
        if not isinstance(response, dict):
            return []

        retrieved_ids = response.get("retrieved_ids")
        if isinstance(retrieved_ids, list):
            return [str(doc_id) for doc_id in retrieved_ids]

        metadata = response.get("metadata", {})
        sources = metadata.get("sources") if isinstance(metadata, dict) else None
        if isinstance(sources, list):
            return [str(doc_id) for doc_id in sources]
        return []

    async def score(self, case: Dict, response: Dict) -> Dict:
        expected_ids = [str(doc_id) for doc_id in case.get("expected_retrieval_ids", [])]
        retrieved_ids = self._extract_retrieved_ids(response)

        hit_rate = self.calculate_hit_rate(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
            top_k=self.top_k,
        )
        mrr = self.calculate_mrr(expected_ids=expected_ids, retrieved_ids=retrieved_ids)

        overlap = self._lexical_overlap(
            answer=response.get("answer", ""),
            expected_answer=case.get("expected_answer", ""),
        )

        faithfulness = min(1.0, max(0.0, 0.25 + 0.5 * overlap + 0.25 * hit_rate))
        relevancy = min(1.0, max(0.0, 0.3 + 0.5 * overlap + 0.2 * hit_rate))

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "retrieval": {
                "hit_rate": round(hit_rate, 4),
                "mrr": round(mrr, 4),
                "top_k": self.top_k,
                "retrieved_ids": retrieved_ids,
                "expected_ids": expected_ids,
            },
        }

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        TODO: Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        if expected_ids is None or retrieved_ids is None:
            return 0.0
        if top_k <= 0:
            return 0.0
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        TODO: Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        if expected_ids is None or retrieved_ids is None:
            return 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có trường 'expected_retrieval_ids' và Agent trả về 'retrieved_ids'.
        """
        if dataset is None or len(dataset) == 0:
            return {
                "total_cases": 0,
                "avg_hit_rate": 0.0,
                "avg_mrr": 0.0,
                "per_case_metrics": []
            }

        per_case_metrics: List[Dict] = []
        total_hit_rate = 0.0
        total_mrr = 0.0

        for case in dataset:
            expected_ids = case.get("expected_retrieval_ids", [])
            retrieved_ids = case.get("retrieved_ids", [])

            hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)

            total_hit_rate += hit_rate
            total_mrr += mrr

            per_case_metrics.append(
                {
                    "question": case.get("question", ""),
                    "hit_rate": hit_rate,
                    "mrr": mrr,
                }
            )

        total_cases = len(dataset)
        return {
            "total_cases": total_cases,
            "avg_hit_rate": total_hit_rate / total_cases,
            "avg_mrr": total_mrr / total_cases,
            "per_case_metrics": per_case_metrics,
        }


async def _quick_test() -> None:
    evaluator = RetrievalEvaluator()

    # Case 1: hit ở vị trí đầu tiên.
    assert evaluator.calculate_hit_rate(["doc_1"], ["doc_1", "doc_2"], top_k=3) == 1.0
    assert evaluator.calculate_mrr(["doc_1"], ["doc_1", "doc_2"]) == 1.0

    # Case 2: có tài liệu đúng nhưng ngoài top_k.
    assert evaluator.calculate_hit_rate(["doc_3"], ["doc_1", "doc_2", "doc_3"], top_k=2) == 0.0
    assert evaluator.calculate_mrr(["doc_3"], ["doc_1", "doc_2", "doc_3"]) == (1.0 / 3.0)

    # Case 3: không có tài liệu đúng.
    assert evaluator.calculate_hit_rate(["doc_x"], ["doc_1", "doc_2"], top_k=3) == 0.0
    assert evaluator.calculate_mrr(["doc_x"], ["doc_1", "doc_2"]) == 0.0

    # Case 4: biên top_k <= 0 và input rỗng.
    assert evaluator.calculate_hit_rate(["doc_1"], ["doc_1"], top_k=0) == 0.0
    assert evaluator.calculate_hit_rate([], ["doc_1"], top_k=3) == 0.0
    assert evaluator.calculate_mrr([], ["doc_1"]) == 0.0

    sample_dataset = [
        {
            "question": "Q1",
            "expected_retrieval_ids": ["doc_1"],
            "retrieved_ids": ["doc_1", "doc_2"],
        },
        {
            "question": "Q2",
            "expected_retrieval_ids": ["doc_3"],
            "retrieved_ids": ["doc_1", "doc_2", "doc_3"],
        },
        {
            "question": "Q3",
            "expected_retrieval_ids": ["doc_x"],
            "retrieved_ids": ["doc_1", "doc_2"],
        },
    ]

    batch_result = await evaluator.evaluate_batch(sample_dataset)
    assert batch_result["total_cases"] == 3
    assert abs(batch_result["avg_hit_rate"] - (2.0 / 3.0)) < 1e-9
    assert abs(batch_result["avg_mrr"] - ((1.0 + (1.0 / 3.0) + 0.0) / 3.0)) < 1e-9
    assert len(batch_result["per_case_metrics"]) == 3

    print("RetrievalEvaluator quick tests passed.")


if __name__ == "__main__":
    asyncio.run(_quick_test())
