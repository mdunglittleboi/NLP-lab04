# Prediction — LAB 04 (viết trước khi chạy experiment)

Dataset: IMDb (25k train / 25k test, review dài khoảng 230 từ, cân bằng 50/50).

> Phần 1 (5 prediction bên dưới) được viết **trước** khi chạy experiment và không chỉnh sửa lại. Phần 2 ở cuối file được thêm **sau** khi có kết quả để đối chiếu.

## Phần 1 — Prediction

### Prediction 1 — TF-IDF + Logistic Regression có hoạt động tốt không?

- **Prediction:** Có, accuracy khoảng 88–90%.
- **Reason:** Sentiment review phim chứa nhiều từ cực tính rõ (excellent, awful, boring…); review dài nên có nhiều tín hiệu từ vựng; TF-IDF giữ nguyên thông tin term, còn Logistic Regression là bộ phân loại tuyến tính rất hợp với không gian sparse chiều cao.
- **Confidence:** Cao (85%).

### Prediction 2 — Average Word2Vec có chắc chắn tốt hơn TF-IDF không?

- **Prediction:** Không. Avg Word2Vec (huấn luyện trên chính tập train) nhiều khả năng thấp hơn TF-IDF vài điểm (khoảng 85–87%).
- **Reason:** Embedding học từ khoảng 20k review là corpus nhỏ; phép trung bình làm mất thứ tự từ và làm loãng các từ cực tính hiếm nhưng quan trọng giữa hàng trăm từ trung tính; một từ chỉ có một vector (không phân biệt ngữ cảnh). Ưu điểm semantic của embedding rõ hơn khi dữ liệu có nhãn ít hoặc text ngắn.
- **Confidence:** Trung bình (65%).

### Prediction 3 — Tăng vocabulary size thì accuracy có chắc chắn tăng không?

- **Prediction:** Không. Accuracy tăng nhanh ở giai đoạn đầu rồi bão hoà (có thể giảm nhẹ).
- **Reason:** Các từ thêm vào về sau là từ hiếm, ít thông tin, dễ gây overfitting/nhiễu; lợi ích giảm dần (diminishing returns), trong khi thời gian và bộ nhớ tăng.
- **Confidence:** Cao (80%).

### Prediction 4 — Bỏ stopwords thì performance có chắc chắn tăng không?

- **Prediction:** Không, thậm chí có thể giảm.
- **Reason:** Danh sách stopwords chuẩn chứa các từ phủ định (not, no, never, nor) và từ nhấn mạnh, vốn rất quan trọng với sentiment. Bỏ chúng làm mất tín hiệu negation. Với Logistic Regression, từ phổ biến vốn đã có trọng số thấp nên lợi ích của việc bỏ stopwords là nhỏ.
- **Confidence:** Trung bình–cao (70%).

### Prediction 5 — Accuracy và F1 có thể cho kết luận khác nhau không?

- **Prediction:** Có, đặc biệt khi dữ liệu mất cân bằng.
- **Reason:** Accuracy bị chi phối bởi class đa số (Bài 4: 95% nhưng F1 positive = 0). Trên IMDb (cân bằng) hai metric sẽ gần nhau, nhưng trên dữ liệu lệch chúng có thể mâu thuẫn.
- **Confidence:** Rất cao (95%).

---

## Phần 2 — Đối chiếu với kết quả thực tế (viết sau khi chạy)

| # | Dự đoán | Kết quả thực tế | Nhận xét |
|---|---|---|---|
| 1 | Accuracy 88–90% | Test: **88.66%** (validation 88.82%), F1 0.8875 | **Đúng.** Nằm trong khoảng dự đoán, sát cận dưới. |
| 2 | Avg Word2Vec thấp hơn TF-IDF, khoảng 85–87% | Avg Word2Vec **84.35%** so với TF-IDF **88.66%** (thấp hơn 4.3 điểm) | **Đúng về hướng**, sai nhẹ về độ lớn: thấp hơn khoảng dự đoán gần 1 điểm. Word2Vec còn chậm hơn nhiều (134.4s so với 7.8s). |
| 3 | Accuracy tăng nhanh rồi bão hoà, có thể giảm nhẹ | max_features 500 → 2,000 → 5,000 → 20,000 → toàn bộ (42,608): val accuracy 0.8306 → 0.8687 → 0.8811 → 0.8874 → 0.8882 | **Đúng về bão hoà** (mức tăng giảm dần: +3.8, +1.2, +0.6, +0.1 điểm). **Sai ở chỗ "có thể giảm"**: không thấy giảm. Thời gian cũng gần như không đổi (8.1–9.7s) nên ý "thời gian tăng" không thể hiện với Logistic Regression + unigram. |
| 4 | Bỏ stopwords không chắc tăng, có thể giảm | Giữ tất cả: val acc 0.8882 / F1 0.8898. Bỏ stopwords: 0.8884 / 0.8904. Bỏ nhưng giữ từ phủ định: 0.8884 / 0.8903 | **Đúng một phần.** Không có cải thiện đáng kể (chênh ≤ 0.0006, tương đương khoảng 1 mẫu trên 4,981 mẫu validation, tức là nhiễu). **Nhưng không giảm** như dự đoán: lập luận "mất tín hiệu phủ định" không thể hiện ở unigram vì các từ như "not" vốn chỉ là một feature yếu. |
| 5 | Accuracy và F1 có thể cho kết luận khác nhau | Tập test lệch 950/50: baseline "luôn đoán negative" có accuracy 0.95, F1 positive = 0, macro-F1 = 0.487. Logistic Regression huấn luyện trên dữ liệu lệch cũng cho accuracy 0.95, precision = recall = F1 = 0 | **Đúng.** Accuracy 95% nhưng F1 positive bằng 0. Điều bất ngờ: Logistic Regression thật cũng "sụp" về lớp đa số (giống hệt baseline) khi chỉ có 50 mẫu positive trên khoảng 1,000 mẫu. |
