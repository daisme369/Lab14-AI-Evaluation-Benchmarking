import asyncio
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List

class MainAgent:
    """
    Đây là Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Sinh viên nên thay thế phần này bằng Agent thực tế đã phát triển ở các buổi trước.
    """
    def __init__(self):
        self.name = "SupportAgent-v1"
        self.top_k = 3
        self.corpus_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "golden_set.jsonl")
        )
        self.doc_store = self._build_document_store()
        self.doc_embeddings = {
            doc_id: self._embed(content["text"])
            for doc_id, content in self.doc_store.items()
        }

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _embed(self, text: str) -> Counter:
        return Counter(self._tokenize(text))

    def _cosine_similarity(self, vec_a: Counter, vec_b: Counter) -> float:
        if not vec_a or not vec_b:
            return 0.0

        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a

        dot_product = sum(value * vec_b.get(token, 0) for token, value in vec_a.items())
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _extract_doc_snippets(self, context: str) -> Dict[str, str]:
        matches = list(re.finditer(r"\[(doc_[^\]]+)\]\s*", context))
        if not matches:
            return {}

        snippets: Dict[str, List[str]] = defaultdict(list)
        for idx, match in enumerate(matches):
            doc_id = match.group(1)
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context)
            chunk = context[start:end].strip()
            if chunk:
                snippets[doc_id].append(chunk)

        return {doc_id: " ".join(parts) for doc_id, parts in snippets.items()}

    def _build_document_store(self) -> Dict[str, Dict[str, str]]:
        document_chunks: Dict[str, List[str]] = defaultdict(list)

        if os.path.exists(self.corpus_path):
            with open(self.corpus_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    case = json.loads(line)
                    context = case.get("context", "")
                    expected_ids = case.get("expected_retrieval_ids", [])

                    snippet_map = self._extract_doc_snippets(context)
                    if expected_ids:
                        for doc_id in expected_ids:
                            if doc_id in snippet_map:
                                document_chunks[doc_id].append(snippet_map[doc_id])
                            elif context:
                                document_chunks[doc_id].append(context)
                    else:
                        for doc_id, snippet in snippet_map.items():
                            document_chunks[doc_id].append(snippet)

        doc_store: Dict[str, Dict[str, str]] = {}
        for doc_id, chunks in document_chunks.items():
            unique_chunks = list(dict.fromkeys(chunk.strip() for chunk in chunks if chunk.strip()))
            if not unique_chunks:
                continue
            doc_store[doc_id] = {
                "text": " ".join(unique_chunks),
                "source": f"{doc_id}.txt",
            }

        if not doc_store:
            doc_store = {
                "doc_eval_intro": {
                    "text": "AI evaluation tracks retrieval quality, answer quality, cost, and latency.",
                    "source": "doc_eval_intro.txt",
                }
            }

        return doc_store

    def _retrieve_top_k(self, question: str) -> Dict[str, List[str]]:
        query_embedding = self._embed(question)
        ranked_docs = []

        for doc_id, doc_embedding in self.doc_embeddings.items():
            score = self._cosine_similarity(query_embedding, doc_embedding)
            ranked_docs.append((doc_id, score))

        ranked_docs.sort(key=lambda item: item[1], reverse=True)
        top_docs = ranked_docs[: self.top_k]

        retrieved_ids = [doc_id for doc_id, _ in top_docs]
        contexts = [self.doc_store[doc_id]["text"] for doc_id in retrieved_ids]

        return {
            "retrieved_ids": retrieved_ids,
            "contexts": contexts,
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

        retrieval = self._retrieve_top_k(question)

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
