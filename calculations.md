# Calculations — LAB 04

## Bài 1 — Binary classification

Cho x = [1, 2, 0], w = [0.5, -0.2, 0.4], b = 0.1.

z = wᵀx + b = 0.5·1 + (−0.2)·2 + 0.4·0 + 0.1 = 0.5 − 0.4 + 0 + 0.1 = **0.2**

P(y=1 | x) = σ(0.2) = 1 / (1 + e^(−0.2)) = 1 / (1 + 0.8187) ≈ **0.5498**

## Bài 2 — Prediction threshold

P(y=1 | x) = 0.72

- τ = 0.5: 0.72 ≥ 0.5 → dự đoán **class 1 (positive)**.
- τ = 0.8: 0.72 < 0.8 → dự đoán **class 0 (negative)**.

**Ảnh hưởng của threshold lên precision và recall:**

- Tăng τ → model khó gán positive hơn → ít dự đoán positive → FP giảm, FN tăng → precision thường tăng, recall giảm.
- Giảm τ → model dễ gán positive hơn → FP tăng, FN giảm → precision thường giảm, recall tăng.
- Hai metric đánh đổi nhau; chọn τ tuỳ chi phí của từng loại lỗi.

## Bài 3 — Confusion matrix

|                     | Pred Positive | Pred Negative |
|---------------------|---------------|---------------|
| **Actual Positive** | TP = 80       | FN = 20       |
| **Actual Negative** | FP = 10       | TN = 90       |

- Precision = TP / (TP + FP) = 80 / 90 ≈ **0.8889**
- Recall = TP / (TP + FN) = 80 / 100 = **0.8**
- F1 = 2PR / (P + R) = (2 · 0.8889 · 0.8) / (0.8889 + 0.8) ≈ **0.8421**
- Accuracy = (80 + 90) / 200 = **0.85**

**Nếu là bài toán phát hiện spam (positive = spam):**

- **False Positive (FP = 10):** email bình thường bị đánh nhầm là spam → người dùng có thể mất email quan trọng (thường là lỗi tốn kém hơn).
- **False Negative (FN = 20):** email spam lọt vào inbox → gây phiền, có thể chứa lừa đảo/phishing.

## Bài 4 — Accuracy trap

Dataset: 950 negative / 50 positive. Model luôn dự đoán negative: TP = 0, FN = 50, FP = 0, TN = 950.

- Accuracy = 950 / 1000 = 95%.
- Recall (positive) = 0 / 50 = 0 → bỏ sót toàn bộ positive.
- Precision (positive) = 0 / 0 → không xác định (quy ước bằng 0).
- F1 (positive) = 0; F1 (negative) ≈ 0.974 → macro-F1 ≈ **0.487**.

**Kết luận:** Model này không thực sự tốt, nó chỉ học "đoán class đa số". Cần xem thêm: confusion matrix, precision/recall/F1 của class thiểu số, macro-F1 (hoặc balanced accuracy), PR-AUC, và so với baseline majority class.
