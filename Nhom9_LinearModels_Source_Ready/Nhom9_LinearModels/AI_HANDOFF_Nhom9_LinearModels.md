# AI HANDOFF – Nhóm 9 Linear Models (Kiến trúc Source Code + JSON)

## 1. Mục tiêu

Hoàn thiện dự án Machine Learning của Nhóm 9 theo yêu cầu giảng viên: giữ Notebook học thuật, đồng thời phải có Source Code Python được chia module rõ ràng. Dự án phải chạy được từ dữ liệu đầu vào đến huấn luyện, đánh giá, lưu mô hình và dự đoán dữ liệu CSV mới.

## 2. Nguồn chuẩn

- Dataset: `insurance.csv` trong gói dự án mới nhất người dùng cung cấp.
- Notebook học thuật chuẩn: nội dung cập nhật từ `Nhom9_LinearModels_Lab.ipynb`, tên trong cấu trúc cuối cùng là `notebooks/Main.ipynb`.
- Đề tài: Regression Algorithms → Linear Models.
- Biến mục tiêu: `charges` (USD).
- 4 thuật toán: Linear Regression (OLS), Ridge, Lasso, ElasticNet.
- Không tự ý đổi dữ liệu, thuật toán, kết quả tham chiếu hoặc cấu hình nhóm nếu chưa phân tích và thống nhất.

## 3. Các quyết định đã được người dùng chốt

1. Dùng scikit-learn cho bốn mô hình chính.
2. Giữ Notebook, refactor để gọi chức năng trong `src/` thay vì lặp lại toàn bộ code chính.
3. Bắt buộc có thư mục `models/` lưu bốn mô hình đã huấn luyện.
4. Không dùng `.pkl`/`.joblib`; lưu mô hình bằng JSON để đọc được trong VS Code/Notepad.
5. `predict.py` nhận CSV đầu vào và xuất CSV dự đoán bằng cả bốn mô hình.
6. `data/processed/` lưu bốn CSV `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` ở dạng đặc trưng gốc đã làm sạch/chia tập; không lưu một bản one-hot/scaled riêng làm nguồn chính.
7. Cross-Validation phải fit encoder/scaler bên trong từng fold; chỉ dùng Train để khảo sát tham số.

## 4. Cấu hình thực nghiệm không được tự ý thay đổi

- `test_size = 0.2`
- `random_state = 42`
- `CV_FOLDS = 5`
- Ridge: `alpha = 10.0`
- Lasso: `alpha = 100.0`, `max_iter = 20000`
- ElasticNet: `alpha = 100.0`, `l1_ratio = 0.9998`, `max_iter = 20000`
- OneHotEncoder: `drop="first"`, `handle_unknown="ignore"`
- StandardScaler: fit trên Train, Test chỉ transform.
- Số dòng: 1.338 gốc → 1.337 sau loại 1 dòng trùng → 1.069 Train / 268 Test.
- Số đặc trưng sau One-Hot: 8.

Alpha=100 của Lasso được dùng để minh họa Feature Selection, không được mô tả là alpha tối ưu tuyệt đối. CV grid trong bản tham chiếu cho alpha tốt nhất khoảng 89.02; chênh lệch CV RMSE so với alpha 100 khoảng 0.06 USD.

ElasticNet với `l1_ratio=0.9998` gần như Lasso, không phải cấu hình cân bằng L1/L2. Cần giải thích đúng điều này khi trình bày.

## 5. Kiến trúc Source Code đã thống nhất

```text
Nhom9_LinearModels/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/insurance.csv
│   ├── processed/X_train.csv
│   ├── processed/X_test.csv
│   ├── processed/y_train.csv
│   ├── processed/y_test.csv
│   └── samples/new_customers_example.csv
├── models/
│   ├── linear_regression.json
│   ├── ridge_model.json
│   ├── lasso_model.json
│   ├── elasticnet_model.json
│   └── preprocessor.json
├── reports/
│   ├── figures/
│   └── tables/
├── docs/
├── notebooks/Main.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── models.py
│   ├── tune.py
│   ├── train.py
│   ├── evaluate.py
│   ├── persistence.py
│   ├── predict.py
│   └── pipeline.py
└── tests/test_json_persistence.py
```

## 6. Trách nhiệm các module

- `config.py`: cấu hình, đường dẫn dự án, tham số mô hình.
- `preprocess.py`: đọc/kiểm tra/làm sạch/chia dữ liệu, lưu processed CSV, One-Hot Encoder và StandardScaler.
- `models.py`: khởi tạo bốn estimator scikit-learn.
- `tune.py`: khảo sát alpha/l1_ratio bằng 5-fold CV trên Train; preprocessing phải nằm trong Pipeline của từng fold.
- `train.py`: huấn luyện bốn estimator.
- `evaluate.py`: MSE, RMSE, MAE, R², bảng trọng số và số biến bằng 0.
- `persistence.py`: lưu/nạp JSON, không sử dụng Pickle/joblib.
- `predict.py`: đọc CSV khách hàng mới, áp dụng JSON preprocessor, dự đoán bằng cả bốn model, xuất CSV.
- `pipeline.py`: chạy toàn bộ flow.
- `notebooks/Main.ipynb`: giữ lý thuyết, công thức, phân tích sâu, biểu đồ và kết luận; import chức năng dùng chung từ `src/`.

Các phần Coordinate Descent tự cài, KKT, Lasso Path, hình học L1/L2, Soft-Thresholding, VIF, residual diagnostics và mô phỏng ElasticNet trong Notebook là nội dung học thuật; không thay thế bốn estimator scikit-learn chính.

## 7. Persistence JSON – yêu cầu kỹ thuật bắt buộc

JSON không lưu nguyên đối tượng scikit-learn. Đây là gói tham số suy luận cho mô hình tuyến tính.

- Mỗi model JSON cần có tên mô hình, lớp estimator, cấu hình, danh sách/ thứ tự đặc trưng, `coefficients`, `intercept`, target và schema version.
- `preprocessor.json` cần lưu danh sách biến số/biến phân loại, categories đã học của OneHotEncoder, quy tắc drop-first, thứ tự 8 đặc trưng, `mean` và `scale` của StandardScaler.
- `predict.py` phải tái tạo đúng OneHotEncoder(drop="first", handle_unknown="ignore") và StandardScaler, sau đó tính `y_hat = intercept + X_scaled @ coefficients`.
- Không được chỉ kiểm tra file JSON có tồn tại. Phải nạp JSON và so sánh dự đoán với scikit-learn, sai khác số học phải rất nhỏ (ngưỡng kiểm tra 1e-6 USD).
- Không nạp JSON từ nguồn không tin cậy nếu có thể chứa dữ liệu đầu vào không an toàn; chỉ dùng JSON do dự án tạo và kiểm tra schema.

## 8. Các lệnh chạy

Từ thư mục gốc repository:

```bash
pip install -r requirements.txt
python -m src.pipeline
python -m src.predict --input data/samples/new_customers_example.csv --output reports/predictions_example.csv
python -m unittest discover -s tests -v
```

Notebook:

```bash
jupyter lab
```

Mở `notebooks/Main.ipynb`, chọn Restart Kernel and Run All.

## 9. Tiêu chí nghiệm thu

- 36 ô code Notebook chạy không lỗi.
- Notebook giữ nội dung học thuật, dùng source modules cho các thao tác chính.
- Chạy `python -m src.pipeline` tạo đủ processed CSV, bảng metrics/CV và 5 JSON.
- Có đúng 4 file model JSON và 1 `preprocessor.json`; không có `.pkl`/`.joblib`.
- `predict.py` nhận CSV và xuất bốn cột dự đoán.
- Dự đoán từ JSON khớp scikit-learn trong sai số 1e-6 USD.
- Test xác nhận cả category lạ (hành vi như handle_unknown=ignore).
- Có đủ 14 biểu đồ PNG và các bảng CSV.
- README/hướng dẫn nêu đúng quy trình, môi trường và những hạn chế cần lưu ý.

## 10. Kết quả tham chiếu

| Model | RMSE Test (USD) | MAE Test (USD) | R² Test | Zero weights |
|---|---:|---:|---:|---:|
| OLS | 5,956.34 | 4,177.05 | 0.8069 | 0 |
| Ridge | 5,985.73 | 4,202.58 | 0.8050 | 0 |
| Lasso | 6,026.54 | 4,202.30 | 0.8024 | 2 |
| ElasticNet | 6,093.51 | 4,261.29 | 0.7979 | 2 |

Cross-validation RMSE mean khoảng 6,121–6,129 USD; SD giữa folds khoảng 210 USD. Không nên nói OLS vượt trội rõ rệt; khác biệt CV giữa mô hình nhỏ hơn nhiều so với độ dao động giữa folds.

## 11. Những điểm cần nói trung thực

- Chỉ số Test cao hơn CV trong dữ liệu này; kết quả Test dựa trên một lần chia ngẫu nhiên.
- Residual có lệch phải và phương sai không đều theo phần phân tích Notebook; mô hình tuyến tính cộng gộp có giới hạn.
- Các hệ số mô tả mối liên hệ thống kê, không chứng minh nhân quả.
- Thí nghiệm ElasticNet group stability là dữ liệu mô phỏng, không phải insurance.csv.
- Số chữ số cuối có thể thay đổi nhẹ giữa phiên bản scikit-learn/NumPy.
- Khi tiếp tục chỉnh sửa, chạy lại pipeline, Notebook và test trước khi giao sản phẩm.
