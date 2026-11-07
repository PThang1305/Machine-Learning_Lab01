# Kết quả thực nghiệm

## 1. Đánh giá trên tập Test

| Mô hình | MSE (USD²) | RMSE (USD) | MAE (USD) | R² | Số trọng số = 0 |
|---|---:|---:|---:|---:|---:|
| Linear Regression (OLS) | 35,478,020.68 | 5,956.34 | 4,177.05 | 0.8069 | 0 |
| Ridge (L2, α=10) | 35,828,958.09 | 5,985.73 | 4,202.58 | 0.8050 | 0 |
| Lasso (L1, α=100) | 36,319,178.48 | 6,026.54 | 4,202.30 | 0.8024 | 2 |
| ElasticNet (α=100, ρ=0.9998) | 37,130,891.34 | 6,093.51 | 4,261.29 | 0.7979 | 2 |

OLS có RMSE/MAE thấp nhất trên lần chia Test này. Tuy nhiên, không nên diễn giải rằng OLS chắc chắn tốt hơn đáng kể so với ba mô hình còn lại.

## 2. 5-fold Cross-Validation trên Train

| Mô hình | CV RMSE mean | SD RMSE | CV MAE mean | CV R² mean | SD R² |
|---|---:|---:|---:|---:|---:|
| OLS | 6,123.65 | 211.65 | 4,221.96 | 0.7228 | 0.0316 |
| Ridge | 6,124.56 | 210.10 | 4,234.60 | 0.7228 | 0.0306 |
| Lasso | 6,121.25 | 210.01 | 4,209.91 | 0.7231 | 0.0301 |
| ElasticNet | 6,128.94 | 210.78 | 4,238.23 | 0.7226 | 0.0285 |

Chênh lệch CV RMSE giữa các mô hình khoảng 7.7 USD, nhỏ hơn nhiều so với độ lệch chuẩn khoảng 210 USD giữa folds. Vì vậy dữ liệu hiện tại chưa cho thấy một mô hình thắng vượt trội rõ ràng.

## 3. Trọng số

Ba trọng số chuẩn hóa có độ lớn lớn nhất ở cả bốn mô hình là `smoker_yes`, `age`, `bmi`. Lasso và ElasticNet đưa `sex_male` và `region_northwest` về 0 theo ngưỡng kiểm tra `1e-4`, giữ lại 6/8 đặc trưng.

Trọng số là mối liên hệ thống kê trong dữ liệu, không chứng minh quan hệ nhân quả.

## 4. Tham số

- Ridge alpha được khảo sát trên dải từ 0.01 đến 10,000 bằng CV trên Train; alpha nhóm chọn là 10.
- Lasso alpha được khảo sát trên lưới 0.1 đến 1,000; điểm CV thấp nhất trong lần chạy tham chiếu khoảng 89.02. Alpha 100 được giữ để minh họa chọn lọc đặc trưng; CV RMSE khoảng 6,121.25 USD, chỉ cao hơn khoảng 0.06 USD so với điểm thấp nhất trong lưới.
- ElasticNet l1_ratio được khảo sát từ 0.05 đến 1.0. Với alpha=100, khi l1_ratio giảm, phần L2 mạnh hơn và kết quả dự đoán suy giảm rõ. Cấu hình l1_ratio=0.9998 gần như Lasso.

Các giá trị tối ưu có thể thay đổi nhẹ theo phiên bản thư viện hoặc cách tạo lưới CV. Hãy dùng CSV tạo ra từ lần chạy hiện tại làm số liệu báo cáo cuối.

## 5. Biểu đồ

Các biểu đồ PNG nằm trong `reports/figures/`. Chúng gồm so sánh RMSE/MAE, R² và số trọng số bằng 0, True vs Predicted, bảng trọng số, hồi quy đơn biến, residual diagnostics, Ridge coefficient path, Lasso path/geometry/soft-thresholding/alpha tuning và ElasticNet l1_ratio/path/group stability.

Lưu ý: biểu đồ mô phỏng nhóm biến tương quan của ElasticNet dùng dữ liệu mô phỏng để minh họa lý thuyết, không phải dữ liệu `insurance.csv`.
