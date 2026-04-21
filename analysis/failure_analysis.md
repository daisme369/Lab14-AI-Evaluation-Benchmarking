# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 52 (Dataset sinh ra từ `synthetic_gen.py`)
- **Tỉ lệ Pass/Fail:** X/Y (Dựa trên lần chạy 10 cases thử nghiệm: Pass Rate đạt 100%, tuy nhiên vẫn có nhiều trường hợp điểm thấp ở ranh giới 3.0 - 3.5)
- **Điểm RAGAS trung bình:**
    - Hit Rate: 0.60
    - Định vị trung bình (MRR): 0.53 
- **Điểm LLM-Judge trung bình:** 4.2 / 5.0
- **Agreement Rate:** ~87.5%

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng (Dự kiến) | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Retrieval Miss | ~40% | Vector DB trượt do câu hỏi mơ hồ, quá ngắn hoặc dùng từ đồng nghĩa phức tạp. |
| Judge Conflict | ~20% | Mô hình nhỏ (Llama 8B) nhận định sai và mâu thuẫn với nhận định của mô hình lớn (Llama 70B). |
| Ambiguous Refusal | ~15% | Không có cơ chế hỏi lại (Clarification) đối với các truy vấn thiếu chủ ngữ. |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất từ Benchmark)

### Case #1: Đánh giá quá khắt khe từ Multi-Judge
1. **Symptom:** Câu hỏi "MRR thường thưởng cho hệ thống điều gì?". Câu trả lời của Agent khá đúng nguyên văn nhưng điểm trung bình bị kéo tuột xuống 3.0.
2. **Why 1:** Giám khảo B (`llama-3.1-8b`) chấm gắt và phản hồi *"Đáp án không chính xác về khái niệm"*, mâu thuẫn với Giám khảo A.
3. **Why 2:** Faithfulness score của RAGAS trả về rất thấp (0.1667).
4. **Why 3:** Lời văn sinh ra của Gemini (Agent) sử dụng cấu trúc ngữ pháp có thể bị các mô hình Judge nhỏ hiểu nhầm là Hallucination.
5. **Root Cause:** Độ nhiễu và thiên kiến (Bias) của các Model cỡ nhỏ làm nhiễu hệ số Agreement và kéo điểm Final Score. Phương pháp RAGAS cũng đo Lexical thay vì Semantic dẫn đến phạt sai câu trúc câu.

### Case #2: Lạc đề do câu hỏi mơ hồ (Ambiguous Case)
1. **Symptom:** Câu hỏi "Nó có cần nhanh hơn không?". Câu trả lời là "Tôi không thể trả lời câu hỏi này...". Điểm: 3.5.
2. **Why 1:** Context nạp vào hoàn toàn không liên quan (Hit rate = 0.0, MRR = 0.0).
3. **Why 2:** Hệ thống Embeddings ưu tiên tìm kiếm các từ khóa chung chung như "nhanh" thay vì phân tích đại từ chỉ định "nó".
4. **Why 3:** Agent bị cứng nhắc tuân thủ nhắc nhở "Chỉ trả lời trong context có sẵn".
5. **Root Cause:** Thiếu bước phân tích hoặc Clarification - hệ thống không có khả năng nhận diện 1 câu hỏi "thiếu ngữ cảnh gốc" để hỏi ngược lại người dùng.

### Case #3: Trượt tài liệu do từ vựng đa nghĩa
1. **Symptom:** Câu hỏi về cách xử lý "Khi hai tài liệu mâu thuẫn ẩn đi tài liệu yếu hơn cho tự tin...". Agent trả về hoàn toàn lạc đề môn Kịp thử hồi quy (Regression Gate). Hit rate = 0.0.
2. **Why 1:** Retriever tra cứu nhầm `doc_release_gate` và `doc_retrieval_metrics` thay vì `doc_conflict_resolution`.
3. **Why 2:** Vector Search mặc định trong Agent quá tập trung vào mật độ từ vựng (BM25 style hoặc naive Cosine dựa trên các keyword) chứ không bắt được sắc thái "xung đột/mâu thuẫn".
4. **Why 3:** Chunking size hoặc cách embedding vector của baseline chưa phân tách rõ ý định (Intent) của người dùng.
5. **Root Cause:** Cần phương pháp Semantic Reranking sau khi Retrieval thay vì chỉ tin tưởng 100% vào khoảng cách Cosine thô sơ ban đầu.

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Cập nhật hàm Retrieval: Thêm một lớp **Reranker (như BGE-Reranker)** để khắc phục tình trạng Hit Rate bị rớt thê thảm hiện tại (hiện = 60%).
- [x] Tùy biến Multi-Judge: Nên thay mô hình 8B bằng một mô hình chuyên lý luận (Reasoning Model cấp trung) hoặc áp đặt quy tắc chấm `CoT (Chain-of-Thought)` chặt hơn cho Judge để hạn chế việc Judge nhỏ chấm láo.
- [x] Cải thiện luồng Agent: Cập nhật System Prompt cho Agent để cho phép đặt câu hỏi làm rõ (Clarification Rule) thay vì cố sức Embed và tìm kiếm một truy vấn mơ hồ.
