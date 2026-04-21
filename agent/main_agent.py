import asyncio
from typing import List, Dict

class MainAgent:
    """
    Đây là Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Sinh viên nên thay thế phần này bằng Agent thực tế đã phát triển ở các buổi trước.
    """
    def __init__(self):
        self.name = "SupportAgent-v1"

    def _select_doc_ids(self, question: str) -> List[str]:
        q = question.lower()

        doc_ids: List[str] = []

        if "mrr" in q or "hit rate" in q or "retrieval" in q:
            doc_ids.append("doc_retrieval_metrics")
        if "faithfulness" in q or "hallucination" in q:
            doc_ids.append("doc_ragas_faithfulness")
        if "relevancy" in q:
            doc_ids.append("doc_ragas_relevancy")
        if "judge" in q or "agreement" in q or "cohen" in q:
            doc_ids.append("doc_multi_judge")
        if "position bias" in q:
            doc_ids.append("doc_position_bias")
        if "async" in q or "batch" in q or "rate limit" in q:
            doc_ids.append("doc_async_runner")
        if "chi phi" in q or "token" in q or "cost" in q:
            doc_ids.append("doc_cost_tracking")
        if "release gate" in q or "deploy" in q or "regression" in q:
            doc_ids.append("doc_release_gate")
        if "5 whys" in q or "failure" in q:
            doc_ids.append("doc_failure_analysis")
        if "mơ hồ" in q or "mo ho" in q or "làm rõ" in q or "lam ro" in q:
            doc_ids.append("doc_ambiguity")
        if "xung dot" in q or "mâu thuẫn" in q or "mau thuan" in q:
            doc_ids.append("doc_conflict_resolution")
        if "chunk" in q:
            doc_ids.append("doc_chunking")
        if "prompt injection" in q or "bo qua" in q or "giả vờ" in q or "gia vo" in q:
            doc_ids.append("doc_prompt_injection")
        if "không biết" in q or "khong biet" in q or "không có" in q or "khong co" in q:
            doc_ids.append("doc_ooc_policy")

        if not doc_ids:
            doc_ids = ["doc_eval_intro", "doc_ooc_policy", "doc_ambiguity"]

        # Giữ unique và chỉ lấy top-3 giống pipeline retrieval thực tế.
        unique_doc_ids = list(dict.fromkeys(doc_ids))
        return unique_doc_ids[:3]

    def _mock_retrieve(self, question: str) -> Dict[str, List[str]]:
        """Giả lập tầng retrieval để trả về doc ids có thứ tự xếp hạng."""
        retrieved_ids = self._select_doc_ids(question)
        return {
            "retrieved_ids": retrieved_ids,
            "contexts": [
                "Ngữ cảnh mô phỏng cho doc id đầu tiên.",
                "Ngữ cảnh mô phỏng cho doc id thứ hai.",
                "Ngữ cảnh mô phỏng cho doc id thứ ba."
            ],
            "sources": [f"{doc_id}.txt" for doc_id in retrieved_ids],
        }

    async def query(self, question: str) -> Dict:
        """
        Mô phỏng quy trình RAG:
        1. Retrieval: Tìm kiếm context liên quan.
        2. Generation: Gọi LLM để sinh câu trả lời.
        """
        # Giả lập độ trễ mạng/LLM
        await asyncio.sleep(0.5)

        retrieval = self._mock_retrieve(question)

        # Giả lập dữ liệu trả về
        return {
            "answer": f"Dựa trên tài liệu hệ thống, tôi xin trả lời câu hỏi '{question}' như sau: [Câu trả lời mẫu].",
            "contexts": retrieval["contexts"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "metadata": {
                "model": "gpt-4o-mini",
                "tokens_used": 150,
                "sources": retrieval["sources"],
            }
        }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(resp)
    asyncio.run(test())
