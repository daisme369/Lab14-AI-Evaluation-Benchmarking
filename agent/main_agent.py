import asyncio
from typing import Dict, List

class MainAgent:
    """
    Đây là Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Sinh viên nên thay thế phần này bằng Agent thực tế đã phát triển ở các buổi trước.
    """
    def __init__(self):
        self.name = "SupportAgent-v1"
        self._doc_keywords = {
            "doc_retrieval_metrics": ["hit rate", "mrr", "retrieval", "top-k"],
            "doc_ragas_faithfulness": ["faithfulness", "hallucination", "grounded"],
            "doc_ragas_relevancy": ["relevancy", "relevance", "tra loi", "cau hoi"],
            "doc_multi_judge": ["judge", "agreement", "consensus", "calibration"],
            "doc_position_bias": ["position bias", "a/b", "b/a"],
            "doc_async_runner": ["async", "batch", "rate limit", "parallel"],
            "doc_cost_tracking": ["cost", "chi phi", "token", "gia"],
            "doc_release_gate": ["release", "rollback", "gate", "delta"],
            "doc_failure_analysis": ["5 whys", "failure", "root cause", "clustering"],
            "doc_prompt_injection": ["prompt injection", "bo qua", "ignore"],
            "doc_ambiguity": ["mo ho", "ambiguous", "clarify", "hoi lai"],
            "doc_conflict_resolution": ["mau thuan", "conflict", "nguon"],
            "doc_ooc_policy": ["khong biet", "out of context", "out-of-context"],
            "doc_chunking": ["chunk", "chunking", "kich thuoc"],
            "doc_eval_intro": ["evaluation", "benchmark", "pipeline"],
        }

    def _retrieve_doc_ids(self, question: str, top_k: int = 3) -> List[str]:
        q = question.lower()
        scored: List[tuple] = []
        for doc_id, keywords in self._doc_keywords.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > 0:
                scored.append((doc_id, score))

        if not scored:
            return ["doc_eval_intro", "doc_retrieval_metrics", "doc_multi_judge"][:top_k]

        scored.sort(key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:top_k]]

    async def query(self, question: str, context: str = "", top_k: int = 3) -> Dict:
        """
        Mô phỏng quy trình RAG:
        1. Retrieval: Tìm kiếm context liên quan.
        2. Generation: Gọi LLM để sinh câu trả lời.
        """
        await asyncio.sleep(0.2)
        retrieved_ids = self._retrieve_doc_ids(question, top_k=top_k)
        context_preview = context.split("\n")[0][:180] if context else "Không có context khả dụng."

        return {
            "answer": (
                f"Dựa trên các tài liệu truy hồi ({', '.join(retrieved_ids)}), "
                f"câu hỏi '{question}' được trả lời như sau. Bằng chứng: {context_preview}"
            ),
            "contexts": [
                context_preview,
                "Thông tin bổ sung từ retrieval heuristic."
            ],
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": "gpt-4o-mini",
                "tokens_used": 150,
                "sources": retrieved_ids,
            }
        }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(resp)
    asyncio.run(test())
