# Reflection Cá Nhân - Tran Trung Hau

## 1) Engineering Contribution

Trong bài lab này, phần mình phụ trách chính là Retrieval Evaluation và tích hợp retrieval vào pipeline benchmark.

- Mình triển khai hoàn chỉnh logic tính Hit Rate và MRR cho từng test case, đồng thời tổng hợp trung bình toàn bộ dataset trong [engine/retrieval_eval.py](engine/retrieval_eval.py).
- Mình thêm kiểm thử nhanh ngay trong module để xác minh các trường hợp đúng, sai, và biên cho top-k, giúp phát hiện lỗi logic sớm trước khi chạy benchmark lớn.
- Mình tích hợp kết quả retrieval vào luồng chấm điểm benchmark trong [main.py](main.py), cụ thể là lấy expected_retrieval_ids từ dataset, lấy retrieved_ids từ output agent, sau đó tính hit_rate và mrr cho từng case.
- Mình thay phần retrieval giả lập dựa trên keyword bằng retrieval chạy dữ liệu thật trong [agent/main_agent.py](agent/main_agent.py): xây corpus từ [data/golden_set.jsonl](data/golden_set.jsonl), tạo embedding dạng bag-of-words, tính cosine similarity, và lấy top-k documents.

Kết quả là hệ thống không còn chỉ mô phỏng retrieval, mà đã có bước truy hồi thực trên dữ liệu lab trước khi tính metric.

## 2) Technical Depth

Trong quá trình làm, mình vận dụng và hiểu rõ các khái niệm cốt lõi sau:

- MRR (Mean Reciprocal Rank): đo chất lượng thứ hạng tài liệu đúng đầu tiên. Tài liệu đúng xuất hiện càng sớm thì điểm càng cao theo công thức 1/rank.
- Hit Rate@k: kiểm tra trong top-k có ít nhất một tài liệu đúng hay không. Metric này phản ánh khả năng phủ đúng tài liệu liên quan.
- Position Bias (ở phần judge): hiểu rằng cùng một cặp câu trả lời nhưng đảo vị trí A/B có thể làm judge cho điểm khác nhau, vì vậy cần kiểm tra tính ổn định của quy trình chấm.
- Trade-off chi phí và chất lượng:
	- Rule-based retrieval cho điểm cao nhưng không phản ánh năng lực truy hồi thật.
	- Retrieval theo embedding cosine phản ánh thực tế hơn, nhưng kết quả benchmark nghiêm ngặt hơn và có thể làm hit_rate giảm trong ngắn hạn.

Mình chọn hướng thứ hai để đảm bảo tính trung thực kỹ thuật của hệ thống eval.

## 3) Problem Solving

Vấn đề lớn nhất mình gặp là hit_rate ban đầu bằng 0 dù pipeline chạy không lỗi.

- Triệu chứng: report vẫn có số liệu nhưng retrieval metric không đạt.
- Phân tích nguyên nhân:
	- expected_retrieval_ids trong dataset dùng hệ id dạng doc_xxx.
	- retrieved_ids từ agent lúc đầu dùng tên khác hệ, nên không khớp khi đối chiếu.
- Hướng xử lý theo từng bước:
	- Chuẩn hóa output agent để luôn trả retrieved_ids rõ ràng.
	- Đồng bộ dữ liệu retrieval với định dạng id của golden set.
	- Nâng cấp retrieval từ mock sang truy hồi thật bằng embedding + cosine để tránh overfit theo luật thủ công.
	- Chạy lại benchmark để xác nhận kết quả sau thay đổi.

Sau khi chuyển sang retrieval dữ liệu thật, metric phản ánh chất lượng truy hồi thực tế thay vì số liệu đẹp do mô phỏng.

## 4) Tự Đánh Giá

- Điểm mình làm tốt:
	- Hoàn thành đúng phần Retrieval Evaluation theo rubric, có cả metric level case và aggregate level.
	- Tạo được kết nối kỹ thuật giữa module agent, evaluator và benchmark runner.
	- Giải quyết được lỗi số liệu sai bằng cách truy nguyên đúng nguyên nhân dữ liệu.
- Điểm cần cải thiện:
	- Cần nâng chất lượng embedding và tiền xử lý tiếng Việt tốt hơn để tăng thêm hit_rate trên các câu hỏi khó.
	- Cần bổ sung thêm so sánh nhiều hướng retrieval để có đánh giá toàn diện hơn về chất lượng và chi phí.

## 5) Kế Hoạch Cải Tiến Cá Nhân

- Bổ sung chuẩn hóa văn bản tiếng Việt tốt hơn trước khi embedding.
- Thử TF-IDF weighted vector hoặc hybrid retrieval để tăng recall cho hard cases.
- Viết thêm test cho các tình huống nhiều ground-truth ids để theo dõi sâu hơn mối quan hệ giữa Hit Rate và MRR.