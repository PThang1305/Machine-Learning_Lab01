# Phương pháp thực hiện

## 1. Bài toán

Dự án dùng dữ liệu Medical Cost Personal (`insurance.csv`) để dự đoán `charges` – chi phí bảo hiểm y tế tính bằng USD. Đây là bài toán hồi quy vì đầu ra là một giá trị số liên tục.

## 2. Dữ liệu đầu vào

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `age` | Số | Tuổi |
| `sex` | Phân loại | Giới tính theo dataset |
| `bmi` | Số | Chỉ số BMI |
| `children` | Số nguyên | Số con được bảo hiểm |
| `smoker` | Phân loại | Tình trạng hút thuốc |
| `region` | Phân loại | Khu vực |
| `charges` | Số | Chi phí bảo hiểm cần dự đoán |

Dataset gốc có 1.338 dòng và 7 cột, không có giá trị thiếu, có 1 dòng trùng hoàn toàn. Sau `drop_duplicates()` còn 1.337 dòng.

## 3. Chia dữ liệu

Dữ liệu được chia bằng `train_test_split(test_size=0.2, random_state=42)`:

- Train: 1.069 dòng.
- Test: 268 dòng.

Train dùng để học mô hình và khảo sát tham số. Test được giữ riêng để đánh giá cuối cùng.

## 4. Tiền xử lý

Quy trình được xây dựng trong `src/preprocess.py`.

1. Ba biến số `age`, `bmi`, `children` được giữ dạng số.
2. Ba biến phân loại `sex`, `smoker`, `region` được chuyển bằng `OneHotEncoder(drop="first", handle_unknown="ignore")`.
3. Sau mã hóa thu được 8 đặc trưng.
4. `StandardScaler` đưa các cột về cùng thang đo.
5. Encoder và scaler của mô hình chính chỉ được fit trên Train; Test chỉ transform.

Bốn file CSV trong `data/processed/` là các tập đã làm sạch và chia Train/Test nhưng vẫn giữ giá trị đầu vào ban đầu, chưa One-Hot Encode hoặc Standardize. Điều này giúp dữ liệu dễ đọc và tránh lưu nhiều phiên bản biến đổi khác nhau.

## 5. Bốn mô hình

| Mô hình | Ý tưởng chính | Siêu tham số nhóm sử dụng |
|---|---|---|
| Linear Regression (OLS) | Tìm trọng số làm tổng bình phương sai số nhỏ nhất | Mặc định |
| Ridge (L2) | Thêm hình phạt bình phương trọng số, làm hệ số nhỏ lại | `alpha=10` |
| Lasso (L1) | Thêm hình phạt trị tuyệt đối, có thể đưa trọng số về 0 | `alpha=100`, `max_iter=20000` |
| ElasticNet | Kết hợp hình phạt L1 và L2 | `alpha=100`, `l1_ratio=0.9998`, `max_iter=20000` |

`alpha` điều chỉnh độ mạnh của hình phạt. `l1_ratio` điều chỉnh tỷ lệ L1 trong ElasticNet: 1 tương ứng Lasso thuần; giá trị càng thấp thì thành phần L2 càng lớn. Với `l1_ratio=0.9998`, ElasticNet trong cấu hình nhóm gần giống Lasso, không phải ví dụ cân bằng L1/L2.

## 6. Khảo sát tham số

Các khảo sát alpha của Ridge/Lasso và `l1_ratio` của ElasticNet thực hiện bằng 5-fold Cross-Validation trên Train. `src/tune.py` tạo Pipeline gồm preprocessor và estimator cho mỗi fold; do đó One-Hot Encoder và StandardScaler được fit lại trên phần train của fold, không sử dụng thông tin từ phần validation.

Giá trị được dùng trong bảng so sánh chính vẫn giữ theo cấu hình nhóm chốt. Việc khảo sát tham số giải thích độ nhạy của mô hình và không tự động thay đổi cấu hình chính.

## 7. Đánh giá

- **MSE:** trung bình bình phương sai số, đơn vị USD².
- **RMSE:** căn bậc hai MSE, đơn vị USD; nhạy với sai số lớn.
- **MAE:** trung bình trị tuyệt đối sai số, đơn vị USD.
- **R²:** mức độ mô hình giải thích biến thiên của mục tiêu so với baseline dự đoán trung bình; không gọi là accuracy.

Bên cạnh Test metrics, nhóm báo cáo mean và SD qua 5 folds CV. Không nên kết luận một mô hình vượt trội chỉ dựa vào chênh lệch nhỏ của một lần chia Test.

## 8. Lưu mô hình bằng JSON

Do đây là các mô hình tuyến tính, dự đoán có thể tái tạo từ hệ số và intercept:

\[
\hat y = b + \sum_{j=1}^{p} w_j x_{j,\text{scaled}}
\]

`models/*.json` lưu hệ số, intercept, cấu hình và thứ tự đặc trưng. `preprocessor.json` lưu categories của One-Hot Encoder, mean và scale của StandardScaler đã fit trên Train. `src/predict.py` tái tạo bước mã hóa/chuẩn hóa và áp dụng công thức dự đoán. Đây là định dạng tham số phục vụ inference, không phải object sklearn có thể nạp trực tiếp bằng `pickle.load`.
