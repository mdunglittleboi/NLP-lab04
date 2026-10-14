# Reflection — LAB 04

## 1. Đã học được gì về representation?

Cùng một classifier (Logistic Regression), đổi representation làm kết quả thay đổi rõ rệt: TF-IDF unigram đạt accuracy 0.8866, còn Average Word2Vec chỉ đạt 0.8435 (thấp hơn 4.3 điểm) và chậm hơn khoảng 17 lần (134.4s so với 7.8s). TF-IDF là biểu diễn sparse/lexical: giữ thông tin từng term và trọng số IDF nên mạnh khi tín hiệu nằm ở các từ khoá đặc trưng, như *great, excellent* hay *worst, waste*. Avg Word2Vec là dense/semantic nhưng mất thứ tự từ, một từ chỉ có một vector, và phép trung bình làm loãng các từ cực tính giữa hàng trăm từ trung tính của một review dài. Embedding học từ khoảng 20k review cũng là corpus nhỏ. Không có representation "tốt nhất" tuyệt đối; nó phụ thuộc loại dữ liệu (độ dài, domain) và lượng dữ liệu có nhãn.

Một điểm khác: thay classifier ít ảnh hưởng hơn thay representation. Linear SVM trên cùng TF-IDF chỉ chênh khoảng 1 điểm so với Logistic Regression (0.8774 so với 0.8866 trên test), trong khi validation lại cho kết quả ngược (SVM F1 0.8926 so với 0.8898), nên thứ hạng giữa hai classifier này không ổn định.

## 2. Accuracy không đủ

Bài 4 và phần mô phỏng lệch lớp cho thấy accuracy 95% có thể vô nghĩa: cả baseline "luôn đoán negative" lẫn Logistic Regression huấn luyện trên dữ liệu lệch 95/5 đều đạt accuracy 0.95 nhưng F1 positive bằng 0 (macro-F1 chỉ 0.487). Phải xem confusion matrix, precision/recall/F1 theo class và so với baseline đơn giản. Trên IMDb (cân bằng) hai loại lỗi gần đối xứng (FP rate 0.116 và FN rate 0.110 với Model A) nên accuracy và F1 gần nhau, nhưng điều đó chỉ đúng nhờ dữ liệu cân bằng. Loại lỗi nào đắt hơn (FP hay FN) còn tuỳ ứng dụng và quyết định việc chọn threshold.

## 3. Data leakage rất dễ mắc

Chỉ cần fit TF-IDF trước khi split là thông tin của test đã lọt vào vocabulary và IDF. Khi thử fit trên toàn bộ dữ liệu, vocabulary tăng từ 40,815 lên 59,485 từ (18,670 term chỉ có mặt vì đã "nhìn thấy" validation/test). Ngoài ra còn có 123 review xuất hiện ở cả train lẫn test mà ta phải loại ra. Quy tắc: mọi bước học từ dữ liệu (TF-IDF, Word2Vec, scaler…) chỉ fit trên train.

## 4. Hạn chế của representation hiện tại

Bag-of-words và averaged static embeddings không mô hình hoá **thứ tự từ**, **phạm vi phủ định**, **mỉa mai** và **đa nghĩa**. Các thí nghiệm cho thấy cụ thể:

- Với "this movie is good" và "this movie is not good", Avg W2V + LR cho P(positive) giảm từ 0.9223 xuống 0.0895, còn TF-IDF + LR giảm từ 0.7178 xuống 0.4159. Model có phản ứng với "not", nhưng không phải vì hiểu phủ định mà vì "not" (và các từ như *no, nothing*, nằm trong top feature tiêu cực) tương quan với review tiêu cực trong dữ liệu huấn luyện.
- Hệ quả là "The plot was not bad at all" (nghĩa tích cực) bị cả ba model đoán là tiêu cực (P(positive) lần lượt 0.001, 0.032, 0.000), vì "not" kéo xác suất xuống bất kể nó phủ định từ nào.
- Bigram giúp cục bộ: "not good" có trọng số −0.737 trong khi "good" là +1.603 và "very good" là +1.867. Nhưng "not bad at all" vẫn sai, và câu mỉa mai "What a masterpiece... I fell asleep twice" bị cả ba model đoán nghiêng positive (0.525, 0.571, 0.985).
- Một số lỗi không do model hiểu sai từ mà do đoạn review chỉ tóm tắt cốt truyện, hoặc review có cảm xúc hỗn hợp (xem `error_analysis.md`).

Bigram cải thiện F1 từ 0.8875 lên 0.8953 nhưng làm feature space phình từ 42,608 lên 363,229 (gấp khoảng 8.5 lần) và thời gian huấn luyện tăng từ 9.3s lên 23.7s, mà vẫn không giải quyết được phụ thuộc xa.

## 5. Từ representation đến neural NLP

Các giới hạn trên dẫn tới nhu cầu của mô hình xử lý chuỗi (RNN/LSTM/GRU, LAB 05): cập nhật trạng thái theo từng từ nên nhớ được thứ tự và ngữ cảnh. Sau đó là attention/Transformer và contextual embeddings (một từ có vector khác nhau tuỳ ngữ cảnh), để giải quyết phụ thuộc xa, phạm vi phủ định và đa nghĩa.

## 6. Một số điều bất ngờ trong kết quả

- Bỏ stopwords (kể cả các từ phủ định) hầu như không làm thay đổi kết quả (F1 validation 0.8898 → 0.8904), trái với dự đoán rằng nó có thể làm giảm. Lý do hợp lý: với unigram, "not" vốn chỉ là một feature yếu.
- Accuracy tăng nhanh khi vocabulary tăng từ 500 lên 5,000 từ rồi gần như bão hoà (0.8874 ở 20,000 từ so với 0.8882 ở toàn bộ 42,608 từ), và thời gian huấn luyện gần như không đổi với Logistic Regression.
- Một số top feature của Logistic Regression không phải từ cảm xúc đúng nghĩa: *today* (+2.85), *also* (+2.77), *very* (+2.76), dấu *?* (−5.26), *minutes* (−3.54). Điều này cho thấy model học cả các tín hiệu mang tính dataset-specific.

---

# Individual learning check

- **Tại sao accuracy 95% chưa chắc tốt?** Nếu dữ liệu lệch 95/5 thì model luôn đoán class đa số cũng đạt 95% nhưng recall của class thiểu số bằng 0; cần xem confusion matrix, precision/recall/F1.
- **TF-IDF khác average Word2Vec ở điểm nào?** TF-IDF sparse, lexical, theo từng term (chiều = |V|); avg Word2Vec dense, semantic, 100–300 chiều, mất thứ tự, một từ một vector, và phép trung bình làm loãng thông tin.
- **Tại sao "not good" có thể gây lỗi?** Với bag-of-words/avg embedding, "not" và "good" là hai thành phần độc lập được cộng vào; model không biết "not" đảo nghĩa "good". Kết quả là "not" chỉ kéo xác suất về phía negative theo thống kê chung, nên nó sai ở các cụm như "not bad at all".
- **Data leakage xảy ra ở đâu nếu fit TF-IDF trước khi split?** Ở bước `fit`: vocabulary và IDF được tính từ cả test nên thông tin test đi vào feature huấn luyện.
- **Precision và recall khác nhau thế nào?** Precision = trong số dự đoán positive, bao nhiêu đúng (nhạy với FP); Recall = trong số positive thật, bắt được bao nhiêu (nhạy với FN).
- **Tại sao thêm bigram có thể cải thiện sentiment classification?** Bigram bắt được cụm có nghĩa như "not good", "very good" nên nắm được phủ định/nhấn mạnh cục bộ (F1 từ 0.8875 lên 0.8953).

---

# AI assistance statement

- **Tool:** Claude (Anthropic)
- **Purpose:** Hỗ trợ viết code pipeline (preprocessing, TF-IDF, Word2Vec, evaluation, visualization), debug lỗi khi chạy notebook (tải dataset, tốc độ Word2Vec), cấu trúc notebook và tài liệu. Các phần prediction, calculation, diễn giải metric, phân tích lỗi, reflection và individual learning check được AI soạn nháp với sự cho phép của giảng viên.
- **Generated:** Khung notebook, code, và bản nháp các file `.md`.
- **Modified:** Sửa tên dataset thành `stanfordnlp/imdb` để tải được trên Colab, bỏ comment dòng `pip install`, điền họ tên và MSSV, điền số liệu thật từ kết quả chạy vào các file `.md`.
- **Verified:** Đã chạy toàn bộ notebook trên Google Colab với dữ liệu đầy đủ; các phép tính Bài 1–4 được kiểm tra lại bằng code trong notebook; số liệu trong các file `.md` được đối chiếu với output của notebook.
