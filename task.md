Mình sẽ chuyển README thành một checklist triển khai theo từng bước, có tiêu chí hoàn thành rõ ràng để team bám vào và chốt đủ deliverable nộp bài. Mình sẽ bám sát các mục trong README và bổ sung thứ tự thực thi để tránh thiếu hạng mục quan trọng.Dưới đây là task list step by step để team adapt đầy đủ deliverable theo README, bám trực tiếp các yêu cầu ở README.md, README.md, README.md.

**Mục tiêu**
- Đạt đủ bộ nộp bài: source code, benchmark reports, failure analysis, individual reflections theo README.md.
- Đạt các tiêu chí expert: retrieval, multi-judge reliability, async plus cost, root cause analysis theo README.md.

**Step 0: Chốt baseline và phân vai (15-20 phút)**
1. Freeze baseline chạy hiện tại và lưu artifact hiện có.
2. Chia vai tối thiểu:
3. Data owner: dataset plus retrieval labels.
4. AI backend owner: judge plus runner integration.
5. Analyst owner: failure clustering plus 5 whys report.
6. PM owner: release gate plus checklist nộp bài.
7. Definition of Done:
8. Mỗi owner có 1 đầu việc có file output rõ ràng.

**Step 1: Hoàn thiện Deliverable Giai đoạn 1 (Dataset plus SDG)**
1. Chuẩn hóa schema từng case từ ``synthetic_gen.py`` và bảo đảm có expected retrieval ids ở ``synthetic_gen.py`.`
2. Đảm bảo số lượng lớn hơn hoặc bằng 50, hiện target 52 ở `synthetic_gen.py`.
3. Đảm bảo đủ hard cases:
4. adversarial ở `synthetic_gen.py`
5. out of context ở `synthetic_gen.py`
6. ambiguous/conflict ở `synthetic_gen.py`
7. Re-generate dataset và spot-check 10 case ngẫu nhiên.
8. Definition of Done:
9. `golden_set.jsonl` có từ 50 case.
10. Mỗi case có question, expected answer, context, expected retrieval ids, metadata.

**Step 2: Hoàn thiện Deliverable Giai đoạn 2 (Eval Engine plus Async)**
1. Tích hợp judge thật `llm_judge.py` vào luồng benchmark chính thay vì mock ở `main.py` và `main.py`.
2. Đổi wiring runner ở `main.py` sang dùng module thật.
3. Giữ async batching từ `runner.py` và `runner.py`.
4. Nối retrieval evaluation thực:
5. Dùng logic từ retrieval_eval.py và retrieval_eval.py.
6. Lấy retrieved ids từ response runtime thay vì hardcoded.
7. Bổ sung metrics chi phí:
8. Tổng token usage, ước tính cost per run, cost per case vào summary.
9. Definition of Done:
10. Benchmark chạy async ổn định.
11. Summary có avg score, hit rate, agreement rate, token/cost.
12. Kết quả không còn pattern score cố định toàn bộ case.

**Step 3: Hoàn thiện Deliverable Giai đoạn 3 (Benchmark plus Failure Analysis plus 5 Whys)**
1. Chạy benchmark full dataset và xuất lại:
2. summary.json
3. benchmark_results.json
4. Tạo failure clustering thực từ benchmark results:
5. Nhóm lỗi tối thiểu: hallucination, retrieval miss, incomplete, tone mismatch, ambiguity/conflict.
6. Điền báo cáo thật vào failure_analysis.md, thay toàn bộ placeholder ở failure_analysis.md.
7. Viết 5 Whys cho 3 case tệ nhất, chỉ ra root cause thuộc ingestion/chunking/retrieval/prompting theo README.md.
8. Definition of Done:
9. Failure report không còn X/Y, 0.XX, X.X.
10. Có số liệu cụ thể pass/fail, cluster count, action plan có owner và deadline.

**Step 4: Regression Gate và quyết định Release**
1. So sánh V1 vs V2 thực, không dùng cùng pipeline logic.
2. Mở rộng gate ở `main.py` và `main.py`:
3. Không chỉ avg score, thêm latency threshold và cost threshold.
4. Kết luận rõ Release hoặc Rollback theo rule.
5. Definition of Done:
6. Có delta analysis định lượng.
7. Có rule auto-gate và lý do quyết định.

**Step 5: Chốt Submission Checklist**
1. Đảm bảo 4 nhóm deliverable theo README.md:
2. Source code đầy đủ.
3. Reports đầy đủ.
4. Group report đầy đủ.
5. Individual reflections theo README.md.
6. Chạy check script ở check_lab.py.
7. Đề xuất thêm một pre-submit check nội bộ để bắt placeholder trong failure report.
8. Definition of Done:
9. Không thiếu file.
10. Nội dung report là dữ liệu thật, không template.

**Ưu tiên thực thi trong 1 ngày**
1. Sáng: Step 1 plus Step 2 integration.
2. Trưa: benchmark full plus summary.
3. Chiều: Step 3 report plus Step 4 gate.
4. Cuối ngày: Step 5 final check và đóng gói nộp.

Nếu bạn muốn, mình có thể làm luôn bản Task Board dạng file markdown trong repo với cột Owner, ETA, Status để team tick trực tiếp.