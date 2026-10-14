# Error analysis — LAB 04

Dataset IMDb, tập test 24,678 mẫu. Model được phân tích: **Model A (TF-IDF unigram + Logistic Regression)**, accuracy 0.8866.

## 1. Phân tích confusion matrix

| Model | TP | TN | FP | FN | FP rate | FN rate |
|---|---|---|---|---|---|---|
| A: TF-IDF + LR | 11,042 | 10,838 | 1,428 | 1,370 | 0.116 | 0.110 |
| B: Avg W2V + LR | 10,455 | 10,361 | 1,905 | 1,957 | 0.155 | 0.158 |
| C: TF-IDF + SVM | 10,827 | 10,825 | 1,441 | 1,585 | 0.117 | 0.128 |

(Hình: `confusion_matrix.png`. Test có 12,266 negative và 12,412 positive.)

**1. Class nào bị nhầm nhiều nhất?**
Không có class nào bị nhầm lệch rõ rệt. Với Model A, review negative bị đoán thành positive nhiều hơn một chút (FP rate 0.116 so với FN rate 0.110). Với Model B và C thì ngược lại: review positive bị đoán thành negative nhiều hơn (B: 0.158 so với 0.155; C: 0.128 so với 0.117). Độ chênh giữa hai loại lỗi chỉ từ 0.3 đến 1.1 điểm phần trăm.

**2. Loại error nào chiếm nhiều?**
Model A: FP nhiều hơn FN 58 mẫu (1,428 so với 1,370, tức là chênh khoảng 0.2% tổng test). Model B: FN nhiều hơn FP 52 mẫu. Model C: FN nhiều hơn FP 144 mẫu. Cả hai loại lỗi đều chiếm khoảng 5.5–8% tổng số mẫu và gần như đối xứng, phù hợp với việc dữ liệu cân bằng 50/50.

**3. Metric nào phản ánh vấn đề đó?**
Hướng của độ lệch hiện rõ qua precision và recall của class positive:

- Model A có FP > FN nên **precision (0.8855) < recall (0.8896)**.
- Model C có FN > FP nên **precision (0.8825) > recall (0.8723)**, chênh nhiều nhất trong ba model.
- Model B có FN hơi nhiều hơn FP nên precision (0.8459) hơi cao hơn recall (0.8423).

Vì dữ liệu cân bằng và lỗi gần đối xứng nên accuracy, F1 và macro-F1 gần như trùng nhau (A: 0.8866 và 0.8875). Việc chọn metric nào không làm thay đổi kết luận so sánh giữa các model.

## 2. Phân tích từng mẫu (Model A)

Văn bản dưới đây được rút gọn (350 ký tự đầu của review gốc). P(pos) là xác suất positive Model A dự đoán.

### 3 mẫu phân loại đúng

| # | Text (rút gọn) | True | Predicted | P(pos) | Analysis |
|---|---|---|---|---|---|
| 1 | "Just got back from the European Premiere of The Gamers: Dorkness Rising… it's still a great deal of fun… the acting is certainly not Oscar-worthy, but… it adds to the charm…" | positive | positive | 0.890 | Đúng nhờ các từ cực tính dương rõ (*great, fun, charm*). Review có cụm phủ định "not Oscar-worthy" nhưng tín hiệu dương chiếm ưu thế nên model vẫn đúng. |
| 2 | "Don't listen to most of these people. ill give you a better review of this movie which me and my friend love!…" | positive | positive | 0.893 | Đúng nhờ *better, love*. Review còn trích cụm "the worst acting" của người khác nhưng không đủ làm model nghiêng sang negative. |
| 3 | "It seems that there is great potential for the story line of this film… The acting was flat and the story lacked depth… I have to agree with the other negative comments. I only wish I had read them before buying the DVD…" | negative | negative | 0.342 | Đúng nhờ các từ tiêu cực (*flat, lacked, negative, wish*) dù có "great potential" ở đầu. Xác suất 0.342 gần ngưỡng 0.5 nên model chỉ đúng với độ tự tin thấp. |

### 6 mẫu phân loại sai

| # | Text (rút gọn) | True | Predicted | P(pos) | Category | Analysis |
|---|---|---|---|---|---|---|
| 4 | "This is definitely one of the best Kung fu movies in the history of Cinema. The screenplay is really well done… Chuck… is a great actor. The final fight… is a masterpiece!…" | negative | positive | 0.987 | Nghi vấn nhãn sai (data noise) | Phần văn bản hiển thị hoàn toàn tích cực (*best, well done, great, masterpiece*) nên model đoán positive là hợp lý; đây là một trong những mẫu model sai mà rất tự tin. Có thể phần sau của review đảo ngược, hoặc nhãn gốc không khớp nội dung. Cần đọc toàn bộ review để kết luận. |
| 5 | "When the movie first started I thought cheesy. The first ten minutes were really boring. After the slow beginning and some of the soap opera antics, I started liking it… Now, was it a horror? Not really. It shouldn't have been classified as a horror…" | positive | negative | 0.015 | Mixed sentiment, negation, domain-specific term | Review mở đầu bằng nhiều từ tiêu cực (*cheesy, boring, slow*) rồi mới chuyển sang tích cực ("I started liking it"). Bag-of-words cộng dồn tín hiệu nên các từ tiêu cực ở đầu lấn át. Thêm vào đó "Not really", "shouldn't" là phủ định mà TF-IDF không xử lý được, và "horror" là từ thể loại mang sắc thái tiêu cực nhưng không có nghĩa là chê. |
| 6 | "Mickey Rourke hunts Diane Lane in Elmore Leonard's Killshot. It is not like Mickey Rourke ever really disappeared… He was memorable in: Domino, Sin City, Man on Fire… But in his powerful dramatic performance in The Wrestler (2008)…" | negative | positive | 0.979 | Mixed sentiment, domain-specific term | Review khen các phim và vai diễn khác của diễn viên (*memorable, powerful*) làm nền để so sánh, còn nhận xét về Killshot (từ "But" trở đi) nằm sau. Các từ khen xuất hiện ở đoạn đầu kéo xác suất lên 0.979. Tên phim, tên người là từ không mang thông tin sentiment nhưng làm loãng feature. |
| 7 | "If you have never read the classic science fiction novel this mini-series is based on, it may actually be good. Unfortunately, if you ARE a fan of the book, you probably won't be able to watch more than the first hour or two…" | negative | positive | 0.656 | Negation, mixed sentiment (câu điều kiện) | Từ "good" nằm trong mệnh đề giả định ("it may actually be good") còn lời chê thực sự nằm ở mệnh đề sau ("Unfortunately", "won't be able to watch"). Model không phân biệt được mệnh đề điều kiện với nhận xét thật. Xác suất 0.656 cho thấy model chỉ nghiêng nhẹ. |
| 8 | "It's too bad iameracing wants to deny the reality of Faulkner's and Clarence Brown's purpose in the creating of the story and film of 'Intruder In The Dust'. I suppose the burden of a history of racism is difficult for any Southerner to bear… But to say that this film was not specifically about racism is ridiculous…" | positive | negative | 0.377 | Domain-specific term, negation | Đây là bình luận phản bác một người dùng khác chứ không phải đánh giá phim. Các từ như *too bad, racism, difficult, ridiculous, burden* mang sắc thái tiêu cực nhưng mô tả chủ đề/tranh luận, không phải ý kiến về chất lượng phim. Cụm "not specifically" cũng là phủ định không được xử lý. |
| 9 | "Meek and mild Edward G. Robinson (as Wilbert Winkle) decides to quit his bank job and do what he wants, open a 'fix-it' repair shop behind his house… he has befriended local orphanage resident Ted Donaldson… Young Donaldson is an eager workshop assistant…" | negative | positive | 0.657 | Plot summary, domain-specific term | Phần hiển thị chỉ là tóm tắt cốt truyện với những từ nghe tích cực (*mild, befriended, eager, father figure*), chưa có nhận xét nào. Model dựa vào từ vựng bề mặt nên đoán positive; lời chê (nếu có) nằm ở phần sau mà ta không thấy trong đoạn rút gọn. |

> Ghi chú về cột `auto-flags` trong notebook: cờ "data noise (HTML)" không phản ánh lỗi thật vì bước `clean_text` đã xoá thẻ `<br />` trước khi tokenize; cờ "negation?" bật cho hầu hết review dài (chỉ cần có một từ "not" hoặc "n't") nên không đủ để phân biệt. Vì vậy các category ở bảng trên là do đọc nội dung chứ không dựa vào cờ tự động.

## 3. Tỷ lệ lỗi theo độ dài văn bản

| Số từ | n | error_rate |
|---|---|---|
| (0, 50] | 646 | 0.102 |
| (50, 100] | 2,450 | 0.109 |
| (100, 200] | 11,627 | 0.112 |
| (200, 400] | 6,863 | 0.118 |
| >400 | 3,092 | 0.114 |

Tỷ lệ lỗi gần như phẳng (10–12%) ở mọi khoảng độ dài, thấp nhất ở nhóm ngắn nhất (10.2%) và cao nhất ở nhóm 200–400 từ (11.8%). Vì vậy **"short text" không phải nguyên nhân chính** của lỗi trên IMDb; review dài chỉ hơi khó hơn một chút vì có nhiều từ nhiễu và cảm xúc hỗn hợp hơn.

## 4. Tổng kết nguyên nhân lỗi

- **Mixed sentiment / chuyển cảm xúc trong review** (mẫu 5, 6, 7): bag-of-words cộng dồn mọi từ cực tính, không biết từ nào là kết luận chính.
- **Review chỉ tóm tắt cốt truyện hoặc bình luận về người khác** (mẫu 8, 9): thiếu nhận xét trực tiếp nên model dựa vào các từ bề mặt.
- **Domain-specific term**: tên người, tên phim, từ thể loại (*horror*, *racism*) mang sắc thái nhưng không phản ánh ý kiến về phim.
- **Negation**: các cụm như "Not really", "won't be able to", "not specifically" không được xử lý đúng vì "not" chỉ là một feature độc lập.
- **Nhãn có thể sai** (mẫu 4): model sai nhưng rất tự tin, đáng ngờ là lỗi ở dữ liệu chứ không phải ở model.
- **Cách cải thiện:** bigram (đã thử: F1 tăng từ 0.8875 lên 0.8953), các mô hình chuỗi và contextual embeddings (LAB 05 trở đi) để hiểu thứ tự và phạm vi phủ định.
