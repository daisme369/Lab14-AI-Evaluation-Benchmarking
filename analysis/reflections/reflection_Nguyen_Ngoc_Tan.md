# Báo cáo Cá nhân (Individual Reflection)

- **Họ và tên:** Nguyễn Ngọc Tân
- **Lab:** Day 14 — AI Evaluation Factory (Team Edition)
- **Ngày nộp:** 2026-04-21
- **Repository:** https://github.com/daisme369/Lab14-AI-Evaluation-Benchmarking.git
- **Nhánh cá nhân:** `tan`

---

## 1. Vai trò & Đóng góp trong nhóm (Engineering Contribution)

Trong bài lab này, tôi đảm nhận vai trò **Data Engineer — phụ trách Golden Dataset & Synthetic Data Generation (SDG)**. Đây là tầng nền của toàn bộ hệ thống Evaluation Factory: nếu dataset sai/lệch/thiếu, tất cả metrics phía sau (Hit Rate, MRR, Judge Score) đều vô nghĩa.

### 1.1. Các đóng góp cụ thể

| Hạng mục                    | Mô tả công việc                                                                                                             | File liên quan                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Golden Dataset curation** | Rà soát, chuẩn hóa schema, đảm bảo mỗi case có đủ `question`, `ground_truth`, `ground_truth_doc_ids` để tính Hit Rate & MRR | `data/golden_set.jsonl` (52 cases) |
| **SDG pipeline**            | Tinh chỉnh prompt sinh câu hỏi tự động, thêm nhiễu ngữ pháp (typo, từ đồng nghĩa) để tăng độ khó                            | `data/synthetic_gen.py`            |


### 1.2. Git commits chính (nhánh `tan`)

- `feat(data): expand golden_set to 52 cases with ground_truth_doc_ids`
- `feat(data): add red-teaming hard cases (table-lookup, ambiguous pronoun, out-of-scope)`
- `fix(sdg): normalize diacritics & strip duplicates in synthetic_gen.py`
- `docs(data): add HARD_CASES_GUIDE with taxonomy of failure modes`

> Ghi chú: Mapping commit hash sẽ được cập nhật sau khi push lên nhánh `tan`.

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

## 3. Technical Depth — Giải thích các khái niệm cốt lõi

### 3.1. MRR (Mean Reciprocal Rank)

**Công thức:**

$$ MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i} $$

Trong đó `rank_i` là vị trí (1-indexed) của tài liệu đúng đầu tiên xuất hiện trong top-k kết quả retrieve; nếu không có thì `1/rank_i = 0`.

**Khác gì Hit Rate?**
- **Hit Rate** chỉ quan tâm "có/không" tài liệu đúng nằm trong top-k → *binary*.
- **MRR** thưởng hệ thống nào đẩy tài liệu đúng lên **vị trí đầu**. Ví dụ: 2 hệ thống cùng Hit Rate = 1.0 nhưng hệ thống A đưa doc đúng ở rank 1, hệ thống B ở rank 5 → A có MRR = 1.0, B có MRR = 0.2.
- Trong bài lab này, Hit Rate = 0.67 **chưa đủ** để kết luận; cần đọc kèm MRR để biết liệu khi retriever "trúng" thì có trúng ở top-1 hay nằm lẫn ở cuối list.

### 3.2. Cohen's Kappa — tại sao tốt hơn Agreement Rate đơn thuần

**Agreement Rate** (đang dùng, = 0.986) chỉ đo *tỉ lệ 2 Judge cho cùng điểm*. Vấn đề: nếu cả 2 Judge đều có xu hướng cho điểm cao (ví dụ cả hai luôn cho 5/5), agreement rate sẽ **cao một cách giả tạo** do đồng thuận **ngẫu nhiên**.

**Cohen's Kappa** loại bỏ phần đồng thuận ngẫu nhiên:

$$ \kappa = \frac{p_o - p_e}{1 - p_e} $$

- `p_o` = tỉ lệ đồng thuận quan sát được (observed)
- `p_e` = tỉ lệ đồng thuận nếu 2 Judge cho điểm hoàn toàn ngẫu nhiên (expected)

**Ý nghĩa giá trị κ (theo Landis & Koch):**
| κ           | Mức độ đồng thuận |
| ----------- | ----------------- |
| < 0.20      | Slight            |
| 0.21 – 0.40 | Fair              |
| 0.41 – 0.60 | Moderate          |
| 0.61 – 0.80 | Substantial       |
| > 0.80      | Almost perfect    |

→ Đề xuất cho V3: thay `agreement_rate` bằng **Cohen's Kappa** (hoặc dùng cả hai song song) để báo cáo độ tin cậy của Multi-Judge Engine khách quan hơn.

### 3.3. Position Bias trong LLM-as-Judge

**Position Bias** = xu hướng LLM-Judge thiên vị câu trả lời được đặt ở **vị trí đầu tiên** (hoặc vị trí cuối, tùy model) trong prompt khi so sánh A/B.

**Ví dụ minh họa:**
- Judge GPT-4o nhận `[Answer A, Answer B]` → chọn A.
- Đảo lại `[Answer B, Answer A]` → vẫn chọn câu đứng đầu = B.
- Kết luận: Judge không thực sự đánh giá nội dung, mà bị ảnh hưởng bởi vị trí.

**Cách mitigate (áp dụng cho V3):**
1. **Swap & average:** chạy Judge 2 lần với thứ tự đảo ngược, lấy điểm trung bình.
2. **Shuffle seed:** randomize thứ tự A/B cho từng case.
3. **Pointwise thay vì Pairwise:** chấm từng câu trả lời độc lập theo rubric, không so sánh trực tiếp.
4. **Bias Detection Score:** chạy thử nghiệm với 2 câu trả lời giống hệt nhau → nếu Judge chọn lệch đáng kể từ 50/50 → có position bias.

### 3.4. Trade-off Chi phí vs Chất lượng

| Lựa chọn                                                          |         Chất lượng Judge          | Cost / 1000 eval | Ghi chú                                                  |
| ----------------------------------------------------------------- | :-------------------------------: | :--------------: | -------------------------------------------------------- |
| 1 Judge (GPT-4o)                                                  | Medium (bias không kiểm tra được) |       $15        | Rẻ nhất nhưng rủi ro cao — rubric gọi đây là "điểm liệt" |
| 2 Judge (GPT-4o + Claude 3.5)                                     |   **High** (có agreement rate)    |       ~$28       | **Đang dùng** — cân bằng tốt nhất                        |
| 3 Judge (thêm Gemini)                                             |             Very High             |       ~$40       | Overkill cho dataset 52 cases                            |
| 2 Judge + tier-based (Haiku làm judge phụ, chỉ escalate khi lệch) |               High                |       ~$18       | **Đề xuất V3** — giảm ~35% chi phí, giữ agreement rate   |

**Công thức tối ưu chi phí đã đề xuất:**

```
if |score_A - score_B| <= 1:     # 2 judge chính đồng thuận
    final = avg(score_A, score_B)  # skip judge thứ 3
else:                              # lệch lớn → cần tie-breaker
    final = majority_vote(A, B, C)
```

→ Tiết kiệm ~30% chi phí mà không giảm Agreement Rate (vì chỉ ~10% case rơi vào nhánh escalate).

---

## 4. Problem Solving — Các vấn đề kỹ thuật đã xử lý

### 4.1. Hit Rate = 0 trên nhóm câu hỏi tiếng Việt có dấu
- **Symptom:** Nhóm câu hỏi chứa dấu tiếng Việt (`thường`, `đúng`) → retriever không match dù doc có chứa đúng từ.
- **Root cause:** Tokenizer của embedding model normalize NFD ≠ NFC giữa query và doc.
- **Fix:** Chuẩn hóa về NFC ở cả `synthetic_gen.py` khi ghi golden set và `engine/retrieval_eval.py` khi query.

### 4.2. Judge trả về JSON kèm markdown fence khiến parser crash
- **Symptom:** `json.JSONDecodeError` khi parse response từ Claude.
- **Root cause:** Claude hay bọc JSON trong ` ```json ... ``` ` khiến `json.loads` fail.
- **Fix:** Regex strip markdown fence trước khi parse; fallback về `ast.literal_eval` nếu vẫn lỗi; log case để review.

### 4.3. Xung đột điểm số giữa 2 Judge trên case mơ hồ
- **Symptom:** Một số case có `gpt-4o = 5`, `claude-3-5 = 2` → trung bình 3.5 gây nhiễu.
- **Fix:** Tính `agreement_rate` **per-case** (boolean `|score_A - score_B| ≤ 1`), rồi lấy **trung bình có trọng số** theo agreement thay vì trung bình đều. Đã cải thiện tương quan giữa điểm Judge và Hit Rate.

### 4.4. Trùng/gần-trùng test case trong golden set làm điểm bị nhiễu
- **Symptom:** 3 câu hỏi gần giống nhau về ý → retriever trả cùng doc, làm Hit Rate ảo cao.
- **Fix:** Viết script rapid-dedupe dựa trên `difflib.SequenceMatcher > 0.85` và embedding cosine > 0.95; loại bỏ 4 case trùng trước khi benchmark.

### 4.5. Hit Rate thấp trên nhóm câu hỏi dạng bảng/số liệu
- **Symptom:** Các câu hỏi về số liệu trong bảng có Hit Rate chỉ ~0.3.
- **Root cause (truy ngược với nhóm):** Fixed-size chunking cắt ngang bảng, làm mất cấu trúc hàng/cột.
- **Action:** Ghi nhận và đề xuất chuyển sang **Semantic Chunking** + **table-aware parser** cho V3.

---

## 5. Key Learnings

1. **"Retrieval first, Generation later":** nếu không đo Hit Rate & MRR trước, mọi đánh giá Faithfulness/Relevancy đều chỉ đang đo *"LLM trả lời đẹp cỡ nào với context sai"*.
2. **Một Judge là không đủ — và Agreement Rate đơn thuần cũng chưa đủ:** cần tiến lên Cohen's Kappa để loại bỏ đồng thuận ngẫu nhiên.
3. **Position Bias là rủi ro ẩn của LLM-as-Judge:** không test swap order = không có bằng chứng Judge đang đánh giá nội dung thay vì vị trí.
4. **Async runner tiết kiệm chi phí rõ rệt:** chạy song song 52 cases × 2 judge giảm tổng thời gian benchmark gấp nhiều lần so với tuần tự.
5. **5 Whys ép mình đi sâu:** thay vì dừng ở "agent trả lời sai", phương pháp này buộc phải truy ngược về chunking strategy, embedding model hoặc system prompt.
6. **Regression Gate là tư duy sản phẩm:** so sánh V1 vs V2 bằng số, không bằng cảm tính — lần đầu tôi viết logic tự động quyết định Release/Rollback.

---

## 6. Đề xuất cải tiến cho Agent phiên bản kế tiếp (V3)

Ưu tiên theo tác động ước tính:

- [ ] **Semantic Chunking thay cho Fixed-size** + **table-aware parser** — kỳ vọng nâng Hit Rate từ 0.67 → **≥ 0.85**.
- [ ] **Thêm bước Reranking (cross-encoder)** sau khi retrieve top-k — giảm Hallucination cho nhóm câu hỏi dài.
- [ ] **Siết System Prompt:** bắt buộc "Chỉ trả lời dựa trên context; nếu không đủ thông tin, trả lời 'Không có trong tài liệu'".
- [ ] **Thay Agreement Rate → Cohen's Kappa** trong `reports/summary.json`.
- [ ] **Position Bias mitigation:** chạy Judge 2 lần với swap thứ tự rồi lấy trung bình.
- [ ] **Tier-based Judge** (Haiku làm judge phụ, chỉ escalate khi lệch ≥ 2 điểm) → tiết kiệm ~30% chi phí.
- [ ] **Mở rộng Golden Set lên 100+ cases**, phân tầng theo độ khó (easy/medium/hard).

---

## 7. Tự đánh giá (Self-Assessment)

| Tiêu chí                           | Mức độ | Ghi chú                                                        |
| ---------------------------------- | :----: | -------------------------------------------------------------- |
| Hoàn thành nhiệm vụ Golden Dataset |  9/10  | 52 cases có đủ ground-truth IDs, red-teaming, dedupe           |
| Chất lượng dữ liệu                 |  8/10  | Vẫn còn imbalance giữa nhóm dễ/khó, cần phân tầng rõ hơn ở V3  |
| Technical Depth                    |  8/10  | Nắm được MRR, Cohen's Kappa, Position Bias ở mức áp dụng được  |
| Đóng góp vào báo cáo nhóm          |  8/10  | Cung cấp data cho 5 Whys; tham gia viết failure_analysis.md    |
| Học hỏi & chủ động                 |  9/10  | Chủ động tìm hiểu Cohen's Kappa và Position Bias ngoài yêu cầu |
| Làm việc nhóm                      |  9/10  | Giao tiếp rõ ràng, không gây block các thành viên khác         |

---

## 8. Kết luận

Lab 14 cho tôi trải nghiệm đầy đủ nhất về **"đo trước — sửa sau"** trong AI Engineering. Từ vai trò Data Engineer, tôi hiểu rằng **chất lượng của một Evaluation Factory bị chặn trên bởi chất lượng của Golden Dataset**: retriever tốt đến đâu, judge đồng thuận đến đâu cũng vô nghĩa nếu dataset không có ground truth đáng tin. Bước kế tiếp của cá nhân tôi là xây dựng **Golden Set v2** phân tầng theo độ khó, kết hợp Cohen's Kappa và Position-Bias-aware scoring để biến pipeline này thành **CI cho AI Agent** — chạy tự động mỗi khi prompt, index, hoặc chunking strategy thay đổi.
