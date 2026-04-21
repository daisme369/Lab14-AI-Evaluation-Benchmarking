from typing import List, Dict
import asyncio

class RetrievalEvaluator:
    def __init__(self):
        pass

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
    assert abs(batch_result["avg_hit_rate"] - (1.0 / 3.0)) < 1e-9
    assert abs(batch_result["avg_mrr"] - ((1.0 + (1.0 / 3.0) + 0.0) / 3.0)) < 1e-9
    assert len(batch_result["per_case_metrics"]) == 3

    print("RetrievalEvaluator quick tests passed.")


if __name__ == "__main__":
    asyncio.run(_quick_test())
