# Báo cáo Cá nhân (Individual Reflection)

- **Họ và tên:** Nguyễn Ngọc Tân
- **Lab:** Day 14 — AI Evaluation Factory (Team Edition)
- **Ngày nộp:** 2026-04-21
- **Repository:** https://github.com/daisme369/Lab14-AI-Evaluation-Benchmarking.git
- **Nhánh cá nhân:** `tan`

---

## 1. Vai trò & Đóng góp trong nhóm

Trong bài lab này, tôi tham gia với vai trò chính là Golden Dataset

Các đóng góp cụ thể:

| Hạng mục       | Mô tả công việc                                                   | File liên quan                                   |
| -------------- | ----------------------------------------------------------------- | ------------------------------------------------ |
| Golden Dataset | Rà soát & bổ sung hard cases cho bộ `golden_set.jsonl` (52 cases) | `data/golden_set.jsonl`, `data/synthetic_gen.py` |


---

## 2. Kết quả Benchmark (Agent_V2_Optimized)

Số liệu tổng hợp cuối cùng (từ `reports/summary.json`):

- **Tổng số cases:** 52
- **Average Judge Score:** 2.66 / 5.0
- **Hit Rate (Retrieval):** 0.673
- **Agreement Rate giữa 2 Judge:** 0.986

Nhận xét:
- Agreement Rate **rất cao (98.6%)** → 2 Judge đã được calibrate tương đối ổn định, ít xung đột điểm số.
- Hit Rate **0.67** là điểm yếu chính: gần **1/3 số case** retriever không lấy được tài liệu đúng, kéo theo điểm Judge trung bình thấp (2.66).
- Đây chính là dấu hiệu cho thấy **root cause nằm ở tầng Retrieval/Chunking**, không phải ở tầng Generation.

---

## 3. Học được điều gì (Key Learnings)

1. **"Retrieval first, Generation later":** nếu không đo Hit Rate & MRR trước, mọi đánh giá Faithfulness/Relevancy đều chỉ đang đo *"LLM trả lời đẹp cỡ nào với context sai"*.
2. **Một Judge là không đủ:** việc chạy song song 2 model Judge giúp phát hiện được các case mà một Judge thiên vị, và Agreement Rate trở thành chỉ số độ tin cậy rất hữu ích.
3. **Async runner tiết kiệm chi phí rõ rệt:** chạy tuần tự với 52 cases × 2 judge sẽ rất tốn thời gian; async giảm latency tổng thể đáng kể và ổn định cost-per-eval.
4. **5 Whys ép mình đi sâu:** thay vì dừng ở "agent trả lời sai", phương pháp 5 Whys buộc phải truy ngược về chunking strategy, embedding model, hoặc system prompt.
5. **Regression Gate là tư duy sản phẩm:** so sánh V1 vs V2 bằng số, không bằng cảm tính — đây là lần đầu tôi viết logic tự động quyết định Release/Rollback.

---

## 4. Khó khăn & Cách xử lý

| Khó khăn                                                         | Cách xử lý                                                                                      |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Xung đột điểm số giữa 2 Judge trên một số case mơ hồ             | Thêm bước tính `agreement_rate` theo từng case và lấy **trung bình có trọng số** thay vì bỏ qua |
| Hit Rate thấp trên nhóm câu hỏi dạng bảng/số liệu                | Ghi nhận để đề xuất chuyển từ Fixed-size sang Semantic Chunking ở phase sau                     |
| Git push nhánh cá nhân `tan` lần đầu bị lỗi do chưa set upstream | Xử lý bằng `git push -u origin tan`, đã thành công                                              |
| Trùng/gần-trùng các test case trong golden set làm điểm bị nhiễu | Đánh dấu để `check_lab.py` bổ sung rule dedupe ở phiên bản sau                                  |

---

## 5. Đề xuất cải tiến cho Agent phiên bản kế tiếp (V3)

Ưu tiên theo tác động ước tính:

- [ ] **Semantic Chunking thay cho Fixed-size** — kỳ vọng nâng Hit Rate từ 0.67 → **≥ 0.85**.
- [ ] **Thêm bước Reranking (cross-encoder)** sau khi retrieve top-k — giảm Hallucination cho nhóm câu hỏi dài.
- [ ] **Siết System Prompt:** bắt buộc "Chỉ trả lời dựa trên context; nếu không đủ thông tin, trả lời 'Không có trong tài liệu'".
- [ ] **Judge thứ 3 làm tie-breaker** chỉ kích hoạt khi 2 Judge lệch ≥ 2 điểm → tiết kiệm ~30% chi phí eval.
- [ ] **Mở rộng Golden Set lên 100+ cases**, phân nhóm theo độ khó (easy / medium / hard) để đánh giá phân tầng.

---

## 6. Tự đánh giá (Self-Assessment)

| Tiêu chí                      | Mức độ | Ghi chú                                                  |
| ----------------------------- | ------ | -------------------------------------------------------- |
| Hoàn thành nhiệm vụ được giao | 9/10   | Toàn bộ phần Judge Engine & Runner đã chạy ổn            |
| Chất lượng code               | 8/10   | Còn có thể tách module `llm_judge.py` gọn hơn            |
| Đóng góp vào báo cáo nhóm     | 8/10   | Đã hỗ trợ phần số liệu; phần 5 Whys do cả nhóm cùng viết |
| Học hỏi & chủ động            | 9/10   | Chủ động tìm hiểu thêm RAGAS và reranking                |
| Làm việc nhóm                 | 9/10   | Giao tiếp rõ ràng, không gây block các thành viên khác   |

---

## 7. Kết luận

Lab 14 cho tôi trải nghiệm đầy đủ nhất về **"đo trước — sửa sau"** trong AI Engineering. Thay vì cải tiến agent theo cảm tính, giờ tôi đã có một bộ khung đo lường đủ tin cậy (multi-judge + retrieval metrics + regression gate) để ra quyết định release/rollback dựa trên dữ liệu. Bước kế tiếp của cá nhân tôi sẽ là biến pipeline này thành **CI cho AI Agent**, chạy tự động mỗi khi prompt hoặc index thay đổi.


