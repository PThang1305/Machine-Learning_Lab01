# Nhóm 9 – Linear Models: Dự đoán chi phí bảo hiểm

Dự án học phần **Máy học (Machine Learning)**, chủ đề **Regression Algorithms → Linear Models**. Mục tiêu là dự đoán chi phí bảo hiểm y tế (`charges`, USD) bằng bốn mô hình hồi quy: Linear Regression (OLS), Ridge, Lasso và ElasticNet.

Dự án được tổ chức theo hai phần dùng chung một nguồn xử lý:

- `src/`: mã nguồn Python chia module để tiền xử lý, khởi tạo mô hình, huấn luyện, đánh giá, khảo sát tham số, lưu JSON và dự đoán từ CSV.
- `notebooks/Main.ipynb`: tài liệu học thuật/thuyết trình, giữ công thức, giải thích, phân tích chuyên sâu và biểu đồ; các bước xử lý chính import từ `src/` để tránh viết trùng logic.

## 1. Cấu trúc thư mục

```text
Nhom9_LinearModels/
├── README.md
├── AI_HANDOFF_Nhom9_LinearModels.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # insurance.csv gốc
│   ├── processed/            # X_train, X_test, y_train, y_test (tạo khi chạy)
│   └── samples/              # CSV mẫu để thử predict.py
├── models/                   # 4 model JSON + preprocessor.json (tạo khi train)
├── reports/
│   ├── figures/              # 14 biểu đồ PNG
│   └── tables/               # CSV bảng kết quả (tạo khi chạy)
├── docs/                     # methodology, results, source guide, presentation guide
├── notebooks/
│   └── Main.ipynb            # Notebook học thuật và thuyết trình
├── src/
│   ├── config.py             # đường dẫn và cấu hình dùng chung
│   ├── preprocess.py         # đọc/làm sạch/chia dữ liệu, encoder và scaler
│   ├── models.py             # khai báo 4 estimator scikit-learn
│   ├── tune.py               # Cross-Validation và khảo sát tham số
│   ├── train.py              # huấn luyện mô hình
│   ├── evaluate.py           # MSE, RMSE, MAE, R² và trọng số
│   ├── persistence.py        # lưu/đọc thông tin mô hình bằng JSON
│   ├── predict.py            # dự đoán từ CSV bằng các JSON đã lưu
│   └── pipeline.py           # chạy toàn bộ quy trình train/evaluate/export
└── tests/
    └── test_json_persistence.py
```

## 2. Cài đặt môi trường

Khuyến nghị Python 3.10 trở lên.

```bash
python -m venv .venv

# Windows PowerShell / CMD
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Không cần cài extension riêng để mở file JSON. JSON là định dạng văn bản, có thể đọc bằng VS Code hoặc Notepad.

## 3. Chạy source code

Chạy từ thư mục gốc `Nhom9_LinearModels/`:

```bash
python -m src.pipeline
```

Lệnh này thực hiện:

1. Đọc `data/raw/insurance.csv`.
2. Kiểm tra schema và giá trị thiếu, xóa các dòng trùng hoàn toàn.
3. Chia dữ liệu Train/Test theo tỷ lệ 80/20 với `random_state=42`.
4. Lưu 4 file CSV trong `data/processed/`.
5. Fit `OneHotEncoder(drop="first")` và `StandardScaler` trên Train; Test chỉ được transform.
6. Huấn luyện bốn mô hình bằng scikit-learn.
7. Tính MSE, RMSE, MAE, R² và 5-fold Cross-Validation trên Train.
8. Lưu bảng kết quả vào `reports/tables/` và 4 mô hình cùng preprocessor vào `models/*.json`.

`pipeline.py` tạo bảng và JSON. Để tạo/cập nhật toàn bộ 14 biểu đồ, mở Notebook và chọn **Restart Kernel and Run All**.

## 4. Chạy Notebook

```bash
jupyter lab
```

Mở `notebooks/Main.ipynb`, chọn kernel Python đã cài `requirements.txt`, rồi chọn **Restart Kernel and Run All**.

Notebook giữ các phần lý thuyết và thực nghiệm: OLS, Ridge, Lasso, ElasticNet, VIF, Coefficient Path, Coordinate Descent, KKT, Lasso Path, Cross-Validation, ảnh hưởng `l1_ratio`, mô phỏng nhóm biến tương quan và tổng kết.

## 5. Dự đoán từ một CSV mới

CSV đầu vào phải có các cột:

`age, sex, bmi, children, smoker, region`

Sau khi đã chạy `python -m src.pipeline` để tạo JSON, chạy:

```bash
python -m src.predict --input data/samples/new_customers_example.csv --output reports/predictions_example.csv
```

Kết quả giữ các cột đầu vào và thêm bốn cột:

- `predicted_charges_ols`
- `predicted_charges_ridge`
- `predicted_charges_lasso`
- `predicted_charges_elasticnet`

Có thể thay CSV mẫu bằng CSV khách hàng mới theo đúng schema.

## 6. Vì sao dùng JSON thay `.pkl`?

Nhóm chọn lưu JSON để các thông tin mô hình có thể được kiểm tra trực tiếp bằng trình soạn thảo văn bản, không phụ thuộc vào việc giải mã object Pickle.

- Mỗi file model JSON lưu tên mô hình, cấu hình, thứ tự đặc trưng, hệ số `coefficients` và `intercept`.
- `preprocessor.json` lưu categories của One-Hot Encoder và các giá trị `mean`/`scale` đã học từ Train.
- `predict.py` đọc các JSON, áp dụng đúng tiền xử lý rồi tính dự đoán theo công thức tuyến tính.

JSON không phải bản sao object scikit-learn. Đây là **gói tham số suy luận** được thiết kế cho bốn mô hình tuyến tính của dự án. `persistence.py` và `predict.py` có trách nhiệm lưu/đọc gói này.

## 7. Dữ liệu và phương pháp

- Dataset `insurance.csv`: 1.338 dòng, 7 cột; sau khi bỏ 1 dòng trùng còn 1.337 dòng.
- Chia 80/20: 1.069 dòng Train và 268 dòng Test, `random_state=42`.
- Đầu vào gồm 3 biến số (`age`, `bmi`, `children`) và 3 biến phân loại (`sex`, `smoker`, `region`).
- One-Hot Encoding với `drop="first"` tạo 8 đặc trưng; StandardScaler được fit trên Train.
- Cross-Validation dùng 5 folds trên Train. Preprocessor nằm bên trong Pipeline CV nên encoder/scaler được fit lại riêng trong từng fold.
- Tập Test chỉ dùng cho đánh giá cuối cùng và các phân tích residual/diagnostic đã nêu rõ trong Notebook, không dùng để chọn alpha hoặc `l1_ratio`.

## 8. Kết quả tham chiếu

Kết quả Test trong bản Notebook mới nhất (có thể có sai khác rất nhỏ ở các chữ số cuối do phiên bản thư viện):

| Mô hình | RMSE (USD) | MAE (USD) | R² | Số trọng số = 0 |
|---|---:|---:|---:|---:|
| OLS | 5,956.34 | 4,177.05 | 0.8069 | 0 |
| Ridge (α=10) | 5,985.73 | 4,202.58 | 0.8050 | 0 |
| Lasso (α=100) | 6,026.54 | 4,202.30 | 0.8024 | 2 |
| ElasticNet (α=100, ρ=0.9998) | 6,093.51 | 4,261.29 | 0.7979 | 2 |

Không nên kết luận OLS tốt hơn rõ rệt chỉ từ một lần chia Train/Test. 5-fold CV cho thấy RMSE trung bình của bốn mô hình chênh nhau rất ít so với độ lệch chuẩn giữa các folds. Lasso/ElasticNet có giá trị ở việc tạo mô hình gọn hơn; `l1_ratio=0.9998` làm ElasticNet gần với Lasso.

## 9. Kiểm thử JSON

Chạy:

```bash
python -m unittest discover -s tests -v
```

Bài kiểm tra xác nhận dự đoán tính từ JSON gần như trùng với dự đoán của estimator scikit-learn đã huấn luyện, trong giới hạn sai số số học.

## 10. Tài liệu chi tiết

- [Phương pháp thực hiện](docs/methodology.md)
- [Kết quả thực nghiệm](docs/results.md)
- [Hướng dẫn Source Code](docs/source_code_guide.md)
- [Gợi ý thuyết trình](docs/presentation_guide.md)
- [Báo cáo rà soát và kiểm thử](docs/project_review.md)
- [Tài liệu bàn giao cho AI khác](AI_HANDOFF_Nhom9_LinearModels.md)

## 11. Lưu ý

- Chỉ nạp JSON do nhóm tạo ra hoặc từ nguồn đáng tin cậy; không dùng file mô hình không rõ nguồn gốc.
- Các trọng số mô tả mối liên hệ thống kê trong dataset, không chứng minh quan hệ nhân quả.
- Kết quả phụ thuộc vào dữ liệu và phiên bản thư viện. Hãy chạy lại pipeline/Notebook trong môi trường của nhóm trước khi demo.
