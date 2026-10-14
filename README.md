# LAB 04 — Text Classification

- **Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I, 2026
- **Sinh viên:** Nguyễn Mạnh Dũng — **MSSV:** 23001507
- **Dataset:** IMDb (sentiment, binary: negative / positive), 25k train / 25k test
- **Application:** Option A — Sentiment classification
- **Câu hỏi trọng tâm:** Representation nào phù hợp với bài toán nào, tại sao, và model đang sai ở đâu?

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `classification.ipynb` | Notebook chính: inspection, split, data leakage, các model, evaluation, error analysis, weights, application |
| `preprocessing.py` | Hàm `clean_text`, `tokenize`, `avg_vector`, `docs_to_matrix` |
| `results.csv` | Kết quả 4 model trên tập test |
| `confusion_matrix.png` | Confusion matrix của model A, B, C |
| `calculations.md` | Bài tập tính toán 1–4 |
| `prediction.md` | 5 prediction trước experiment, kèm bảng đối chiếu kết quả thực tế |
| `error_analysis.md` | Phân tích confusion matrix, 3 mẫu đúng, 6 mẫu sai |
| `reflection.md` | Reflection, individual learning check, AI assistance statement |

## Dữ liệu và cách chia

- Ban đầu: 25,000 train / 25,000 test, cân bằng 50/50 (12,500 negative / 12,500 positive), không thiếu text/label.
- Có 96 văn bản trùng trong train, 199 trùng trong test, và 123 văn bản xuất hiện ở cả train lẫn test. Tất cả đã được loại bỏ.
- Sau làm sạch: **train = 19,923 / validation = 4,981 / test = 24,678** (validation tách stratified 20% từ train, seed = 42; tỷ lệ positive 0.501 / 0.501 / 0.503).
- Độ dài review: trung bình 233.8 từ, trung vị 174, nhỏ nhất 10, lớn nhất 2,470.
- Tránh data leakage: TF-IDF, Word2Vec và StandardScaler chỉ fit trên TRAIN rồi mới transform validation/test. Khi thử fit TF-IDF trên toàn bộ dữ liệu, vocabulary tăng từ 40,815 lên 59,485 từ, tức 18,670 term chỉ có mặt vì đã "nhìn thấy" validation/test.

## Pipeline và các model

`Raw text → preprocessing → TF-IDF / Avg Word2Vec → Logistic Regression / Linear SVM → Prediction → Evaluation`

| Model | Representation | Classifier |
|---|---|---|
| A (baseline) | TF-IDF unigram | Logistic Regression |
| B | Average Word2Vec (100d, học trên train) | Logistic Regression (có StandardScaler) |
| C | TF-IDF unigram | Linear SVM |
| D | TF-IDF unigram + bigram | Logistic Regression |

## Kết quả (tập test, 24,678 mẫu)

| Representation | Classifier | Accuracy | Precision | Recall | F1 | Vocab / Dim | Thời gian (s) |
|---|---|---|---|---|---|---|---|
| TF-IDF (unigram) | Logistic Regression | 0.8866 | 0.8855 | 0.8896 | 0.8875 | 42,608 | 7.82 |
| Avg Word2Vec | Logistic Regression | 0.8435 | 0.8459 | 0.8423 | 0.8441 | 44,819 từ → 100d | 134.36 |
| TF-IDF (unigram) | Linear SVM | 0.8774 | 0.8825 | 0.8723 | 0.8774 | 42,608 | 0.54 (chỉ phần fit SVM) |
| TF-IDF (uni+bigram) | Logistic Regression | **0.8940** | 0.8895 | **0.9012** | **0.8953** | 363,229 | 23.70 |

## Kết luận chính

- TF-IDF + Logistic Regression đạt 88.66% accuracy, cao hơn Avg Word2Vec 4.3 điểm phần trăm và nhanh hơn khoảng 17 lần (7.8s so với 134.4s). Với review dài, nhiều từ cực tính đặc trưng, TF-IDF giữ được tín hiệu của từng từ, còn phép trung bình của Word2Vec làm loãng chúng.
- Thêm bigram cho kết quả tốt nhất (F1 0.8953, +0.78 điểm so với unigram), đổi lại vocabulary lớn gấp khoảng 8.5 lần (42,608 → 363,229) và thời gian huấn luyện tăng từ 9.3s lên 23.7s.
- Tăng vocabulary size chỉ có lợi đến một ngưỡng: accuracy validation từ 0.8306 (500 từ) lên 0.8811 (5,000 từ) rồi gần như phẳng ở 0.8874–0.8882 (20,000 đến toàn bộ từ).
- Bỏ stopwords không thay đổi đáng kể (±0.0006 F1 trên validation).
- Model vẫn sai ~11% mẫu, chủ yếu ở review có cảm xúc hỗn hợp, review chỉ tóm tắt cốt truyện, và các cấu trúc phủ định như "not bad at all" (xem `error_analysis.md`).

## Cách chạy

1. Mở `classification.ipynb` trên Google Colab hoặc Jupyter.
2. Dòng `!pip install ...` ở cell Setup cài các thư viện cần thiết.
3. Chạy **Run all** từ đầu. Cell `%%writefile` sinh ra `preprocessing.py`, mục Evaluation sinh ra `results.csv` và `confusion_matrix.png`.
4. Có thể đặt `SUBSAMPLE = 5000` ở cell Setup để chạy thử nhanh; đặt `None` để chạy đầy đủ.
5. Dataset được tải bằng `load_dataset("stanfordnlp/imdb")` (tên ngắn `"imdb"` bị từ chối trên Colab mới).
