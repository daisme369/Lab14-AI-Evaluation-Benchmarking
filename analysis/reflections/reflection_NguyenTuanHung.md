# Báo cáo Cá nhân (Individual Reflection)

- **Họ và tên:** Nguyễn Tuấn Hưng - 2A202600230
- **Lab:** Day 14 — AI Evaluation Factory
- **Ngày nộp:** 2026-04-21
- **Repository:** https://github.com/daisme369/Lab14-AI-Evaluation-Benchmarking.git
- **Nhánh cá nhân:** `hungnt`

---

## 1. Vai trò & Đóng góp trong nhóm (Engineering Contribution)

Trong dự án này, tôi đảm nhận vai trò **AI Engineer — phụ trách xây dựng Multi-Judge Evaluation Engine**. Nhiệm vụ chính của tôi là thiết kế bộ khung "trọng tài" AI có khả năng đánh giá câu trả lời của Agent một cách khách quan, ổn định và tự động hóa hoàn toàn.

### 1.1. Các đóng góp cụ thể

| Hạng mục | Mô tả công việc | File liên quan |
| :--- | :--- | :--- |
| **Multi-Judge Engine** | Xây dựng class `LLMJudge` hỗ trợ gọi đồng thời nhiều model (Gemini Direct SDK, Groq/OpenRouter) để chấm điểm chéo. | `engine/llm_judge.py` |
| **Robust Parsing** | Triển khai cơ chế **Regex-based JSON Extraction** để xử lý các phản hồi lỗi hoặc bị format sai (markdown fences) từ LLM. | `engine/llm_judge.py` |
| **Calibration Logic** | Thiết kế thuật toán **Tie-breaker**: tự động gọi Judge thứ 3 (mediator) khi 2 Judge chính lệch nhau > 1 điểm. | `engine/llm_judge.py` |
| **API Integration** | Cấu hình trực tiếp Gemini Google SDK và Groq SDK để tối ưu tốc độ và vượt qua giới hạn rate limit của các bên trung gian. | `engine/llm_judge.py` |
| **Pipeline Integration** | Kết nối Evaluation Engine vào luồng chạy chính của `main.py`, thay thế các class giả lập bằng mô hình thật. | `main.py` |

### 1.2. Các giải pháp kỹ thuật chính đã thực hiện

- `feat(judge): implement dual-judge scoring (Gemini + Groq) with tie-breaker logic`
- `fix(parsing): add robust regex extraction to handle malformed JSON responses`
- `feat(perf): implement async retries with exponential backoff for rate-limit handling`
- `fix(routing): decouple Gemini calls from OpenRouter to direct Google SDK to solve 504 timeouts`

---

## 2. Kết quả Benchmark (Agent_V2_Optimized)

Số liệu thực tế từ lần chạy cuối cùng (`reports/summary.json`):

- **Tổng số cases:** 52
- **Average Judge Score:** 3.11 / 5.0
- **Hit Rate (Retrieval):** 0.673
- **Agreement Rate (Giữa 2 Judge):** 0.796

**Nhận xét:**
- **Agreement Rate (79.6%)**: Mức đồng thuận ở mức khá. Việc sử dụng hai dòng model khác nhau (Gemini và Llama) mang lại cái nhìn đa chiều nhưng cũng dẫn đến sự khác biệt về tiêu chí chuyên nghiệp/an toàn.
- **Judge Score (3.11)**: Điểm số phản ánh đúng thực trạng của Agent. Agent trả lời khá ổn nhưng bị kéo thấp bởi lỗi retrieval.
- **Hit Rate (0.673)**: Đây là nút thắt cổ chai lớn nhất. Khi retriever chỉ lấy đúng tài liệu trong ~67% trường hợp, Judge không thể cho điểm tối đa vì thiếu dữ kiện (hallucination).

---

## 3. Technical Depth — Giải thích các khái niệm cốt lõi

### 3.1. Tại sao cần Multi-Judge Calibration?

Dùng một LLM duy nhất làm Judge (ví dụ chỉ dùng GPT-4o) thường dẫn đến hiện tượng **Self-preference bias** (LLM có xu hướng chấm điểm cao cho chính phong cách trả lời của nó).

**Giải pháp Tie-breaker tôi đã triển khai:**
```python
if abs(score_A - score_B) > 1.0:
    # Gọi thêm Judge thứ 3 làm mediator
    tie_breaker_res = await self.evaluate_single_judge("groq", ...)
    final_score = statistics.median([score_A, score_B, score_C])
```
Cơ chế này giúp loại bỏ các "outlier" khi một Judge bỗng dưng quá khắt khe hoặc quá dễ dãi trên một case cụ thể.

### 3.2. Position Bias trong LLM-as-Judge

LLM thường bị thiên vị bởi vị trí của thông tin trong Prompt. Trong đánh giá Pairwise (so sánh A vs B), LLM có xu hướng chọn câu trả lời đứng ở vị trí A nhiều hơn.

**Cách chúng tôi phòng tránh:**
- Sử dụng **Pointwise Evaluation**: Mỗi Judge đánh giá câu trả lời dựa trên Rubric độc lập, không so sánh trực tiếp hai phương án cùng lúc để giảm áp lực lên cửa sổ ngữ cảnh và tránh thiên vị thứ tự.

### 3.3. Robust JSON Extraction (Regex vs Native Parsing)

Trong môi trường thực tế, API có thể trả về:
- `Here is the result: {"score": 5...}`
- ` ```json {"score": 5...} ``` `

Nếu dùng `json.loads()` trực tiếp, hệ thống sẽ crash. Tôi đã dùng Regex để "trích xuất" nội dung nằm trong cặp ngoặc nhọn `{...}` trước khi parse, giúp hệ thống đạt độ bền bỉ (reliability) cao ngay cả khi model trả về thừa ký tự.

---

## 4. Problem Solving — Các vấn đề kỹ thuật đã xử lý

### 4.1. Lỗi Gemini 404 & 429 qua OpenRouter
- **Vấn đề:** Khi gọi Gemini qua OpenRouter, hệ thống liên tục gặp lỗi 504 Timeout hoặc 429 Quota Exceeded.
- **Xử lý:** Chuyển hoàn toàn sang dùng **Google Generative AI SDK** chính thức. Điều này giúp tận dụng Quota riêng của Google và giảm độ trễ (latency).

### 4.2. Rate Limit trên Groq Free Tier
- **Vấn đề:** Groq giới hạn TPM (Tokens per minute) rất thấp, dẫn đến crash khi chạy batch 5-10 câu hỏi.
- **Xử lý:** Điều chỉnh `batch_size = 1` và thêm `asyncio.sleep(5)` giữa các lần gọi. Đồng thời tích hợp cơ chế **Retry với Exponential Backoff** trong `evaluate_single_judge`.

### 4.3. Lỗi JSON Formatting từ Llama 3
- **Vấn đề:** Llama thỉnh thoảng không tuân thủ strict JSON format.
- **Xử lý:** Ép kiểu qua tham số `response_format={"type": "json_object"}` và bổ sung thêm hướng dẫn "Chỉ trả về JSON duy nhất" vào System Prompt.

---

## 5. Key Learnings

1.  **AI Eval là một hệ thống xác suất:** Không có "đúng tuyệt đối", chỉ có sự đồng thuận giữa các Judge (Agreement Rate).
2.  **Infrastructure quan trọng không kém Model:** Việc cấu hình SDK trực tiếp và xử lý lỗi mạng (retries) chiếm 70% thời gian để có một pipeline ổn định.
3.  **Tư duy Benchmarking:** Mọi thay đổi code đều phải được đo bằng số liệu (Avg Score, Hit Rate) thay vì cảm nhận cá nhân "thấy nó trả lời hay hơn".

---

## 6. Đề xuất cải tiến cho V3

- [ ] **Tích hợp RAGAS Metrics:** Bổ sung các chỉ số tự động như `faithfulness` và `answer_relevancy` để bổ trợ cho điểm của Judge.
- [ ] **Async Batching tối ưu:** Sử dụng `asyncio.Semaphore` để kiểm soát concurrency tốt hơn thay vì chỉ dùng `sleep` cứng.
- [ ] **Small-model Distillation:** Thử nghiệm dùng Llama 3 8B đã được fine-tune để làm Judge nhằm giảm chi phí (Cost optimization).

---

## 7. Tự đánh giá (Self-Assessment)

| Tiêu chí | Mức độ | Ghi chú |
| :--- | :---: | :--- |
| Kỹ thuật Multi-Judge | 9/10 | Xây dựng được hệ thống ổn định, tích hợp model đa dạng. |
| Problem Solving | 9/10 | Xử lý triệt để các lỗi API và JSON format. |
| Technical Depth | 8/10 | Hiểu sâu về calibration và bias trong đánh giá AI. |
| Làm việc nhóm | 9/10 | Tích hợp thành công code vào pipeline chung của nhóm. |

---

## 8. Kết luận

Buổi Lab 14 đã giúp tôi hiểu rõ tầm quan trọng của việc xây dựng "thước đo" chuẩn trước khi xây dựng sản phẩm. Với hệ thống Multi-Judge Engine đã hoàn thiện, chúng tôi có thể tự tin tối ưu hóa Agent V3 dựa trên dữ liệu thực tế thay vì phán đoán cảm tính. Đây là bước đệm quan trọng để đưa sản phẩm AI ra môi trường Production thực tế.
