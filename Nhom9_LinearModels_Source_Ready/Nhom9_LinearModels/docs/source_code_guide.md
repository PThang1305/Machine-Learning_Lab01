# Hướng dẫn đọc Source Code

Mục tiêu của `src/` là chia một quy trình Machine Learning thành các nhiệm vụ nhỏ. Các file không phải các bản code độc lập cạnh tranh nhau; chúng gọi nhau theo một luồng chung.

| File | Nhiệm vụ | Đầu vào chính | Đầu ra chính |
|---|---|---|---|
| `config.py` | Tập trung đường dẫn và cấu hình | Không | Đường dẫn, alpha, seed, số folds |
| `preprocess.py` | Đọc, kiểm tra, làm sạch, chia và mã hóa dữ liệu | `insurance.csv` | Train/Test, preprocessor đã fit |
| `models.py` | Tạo bốn estimator sklearn | Tên mô hình | Model chưa huấn luyện |
| `tune.py` | Đánh giá cấu hình bằng 5-fold CV | Model, Train raw | CV scores trên Train |
| `train.py` | Huấn luyện estimator | Ma trận Train đã xử lý | Bốn model đã fit |
| `evaluate.py` | Đo chất lượng và lấy trọng số | Model, Test đã xử lý | Bảng metrics, bảng hệ số |
| `persistence.py` | Chuyển model/preprocessor thành JSON và đọc lại | Model đã fit, preprocessor | Các file JSON |
| `predict.py` | Dự đoán dữ liệu CSV mới | CSV khách hàng + JSON | CSV có 4 cột dự đoán |
| `pipeline.py` | Gọi các bước theo thứ tự | Dataset gốc | Processed CSV, bảng kết quả, JSON |

## Sơ đồ chạy

```text
insurance.csv
     |
     v
preprocess.py -----> data/processed/*.csv
     |
     +----> models.py ----> train.py ----> fitted estimators
     |                              |
     |                              +----> evaluate.py ----> reports/tables/*.csv
     |                              |
     |                              +----> persistence.py --> models/*.json
     |
     +----> tune.py (5-fold CV trên Train, fit preprocessing trong từng fold)

CSV khách hàng mới + models/*.json
     |
     v
predict.py
     |
     v
CSV chứa dự đoán OLS / Ridge / Lasso / ElasticNet
```

## Cách chạy

Từ thư mục gốc:

```bash
python -m src.pipeline
python -m src.predict --input data/samples/new_customers_example.csv --output reports/predictions_example.csv
python -m unittest discover -s tests -v
```

`Main.ipynb` import các module này cho quy trình chính. Các cell phân tích sâu có thể tiếp tục dùng scikit-learn/NumPy trực tiếp để minh họa thuật toán, ví dụ Coordinate Descent tự cài, KKT, coefficient path và mô phỏng ElasticNet.
