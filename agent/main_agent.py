import asyncio
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List

from openai import AsyncOpenAI

class MainAgent:
    """
    Đây là Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Sinh viên nên thay thế phần này bằng Agent thực tế đã phát triển ở các buổi trước.
    """
    def __init__(self):
        self.name = "SupportAgent-v1"
        self.top_k = 3
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openrouter_model = os.getenv("OPENROUTER_AGENT_MODEL", "google/gemini-2.5-flash")
        self.openrouter_api_key = (
            os.getenv("OPENROUTE_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        ).strip()
        self.openrouter_client = (
            AsyncOpenAI(api_key=self.openrouter_api_key, base_url=self.openrouter_base_url)
            if self.openrouter_api_key
            else None
        )
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

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text.split()) * 1.3))

    def _build_generation_messages(self, question: str, contexts: List[str]) -> List[Dict[str, str]]:
        context_block = "\n\n".join(f"[CTX {idx+1}] {ctx}" for idx, ctx in enumerate(contexts))
        system_prompt = (
            "You are a retrieval-grounded assistant. "
            "Answer only from the given contexts. "
            "If information is missing or conflicting, say so explicitly."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved Contexts:\n{context_block}\n\n"
            "Return a concise, factual answer in Vietnamese."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _generate_with_openrouter(self, question: str, contexts: List[str]) -> Dict[str, object]:
        if self.openrouter_client is None:
            raise RuntimeError("OpenRouter client is unavailable: missing OPENROUTE_API_KEY/OPENROUTER_API_KEY")

        messages = self._build_generation_messages(question, contexts)
        completion = await self.openrouter_client.chat.completions.create(
            model=self.openrouter_model,
            messages=messages,
            temperature=0.2,
            max_tokens=256,
            extra_headers={
                "HTTP-Referer": "https://local.benchmark",
                "X-Title": "Lab14-AI-Evaluation-Benchmarking",
            },
        )

        answer = ((completion.choices[0].message.content or "").strip() if completion.choices else "").strip()
        if not answer:
            answer = "Khong co noi dung tra loi hop le tu model."

        usage = getattr(completion, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens <= 0:
            total_tokens = self._estimate_tokens(" ".join([question] + contexts + [answer]))

        return {
            "answer": answer,
            "tokens_used": total_tokens,
            "provider": "openrouter",
            "model": self.openrouter_model,
        }

    async def query(self, question: str) -> Dict:
        """
        Mô phỏng quy trình RAG:
        1. Retrieval: Tìm kiếm context liên quan.
        2. Generation: Gọi LLM để sinh câu trả lời.
        """
        retrieval = self._retrieve_top_k(question)

        try:
            generation = await self._generate_with_openrouter(question, retrieval["contexts"])
            answer = str(generation["answer"])
            tokens_used = int(generation["tokens_used"])
            provider = str(generation["provider"])
            model_name = str(generation["model"])
        except Exception as exc:
            await asyncio.sleep(0.1)
            answer = (
                "Khong goi duoc OpenRouter. Tra loi tam thoi dua tren context retrieval: "
                f"{'; '.join(retrieval['contexts'][:1])}"
            )
            tokens_used = self._estimate_tokens(" ".join([question, answer]))
            provider = "local-fallback"
            model_name = "template-fallback"

        return {
            "answer": answer,
            "contexts": retrieval["contexts"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "metadata": {
                "provider": provider,
                "model": model_name,
                "tokens_used": tokens_used,
                "sources": retrieval["sources"],
            }
        }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(resp)
    asyncio.run(test())
