# Nhóm 9 · Machine Learning · Linear Models trên `insurance.csv`

Dự án học phần Máy học (Machine Learning), chủ đề **Regression Algorithms → Linear Models**. Dự án dạng **Notebook-first**: mọi thứ chạy trong một Jupyter Notebook duy nhất.

- **Dữ liệu:** `data/insurance.csv` (Medical Cost Personal, Kaggle). Biến mục tiêu `charges` (chi phí bảo hiểm y tế, USD).
- **Bốn mô hình:** Linear Regression (OLS), Ridge (L2), Lasso (L1), ElasticNet (L1 + L2).
- **Cấu hình nhóm thống nhất:** `LinearRegression()`, `Ridge(alpha=10.0)`, `Lasso(alpha=100.0, max_iter=20000)`, `ElasticNet(alpha=100.0, l1_ratio=0.9998, max_iter=20000)`.

## Cấu trúc thư mục

```text
Nhom9_LinearModels/
├── README.md
├── requirements.txt
├── .gitignore
├── data/insurance.csv
├── notebooks/Nhom9_LinearModels_Lab.ipynb
├── docs/
│   ├── methodology.md          # phương pháp, tiền xử lý, cách đánh giá
│   ├── results.md              # bảng số liệu và cách đọc
│   ├── presentation_guide.md   # phân công, mạch thuyết trình, Q&A 
│   ├── project_review.md       # rà soát, điểm chưa xác minh
│   └── images/*.png            # 14 biểu đồ
└── outputs/*.csv, Nhom9_plots.zip
```

## Cách chạy

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                      # mở notebooks/Nhom9_LinearModels_Lab.ipynb, chọn Restart Kernel and Run All
```

Notebook chạy được từ thư mục gốc dự án **hoặc** từ `notebooks/` (tự tìm `data/insurance.csv`). Khi chạy xong, hình lưu vào `docs/images/` và CSV lưu vào `outputs/`. Trên Google Colab, nếu không thấy file dữ liệu, Notebook sẽ tải từ nguồn công khai hoặc hiện nút upload.

Lần chạy đã kiểm: Python 3.12.3, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2. Chưa thử trên phiên bản khác. Con số có thể lệch rất nhỏ (vài chữ số thập phân của MSE) nếu dùng phiên bản scikit-learn khác.

## Dữ liệu và cách chia

- Dữ liệu gốc 1.338 dòng, 7 cột; không có giá trị thiếu, có 1 dòng trùng. Sau `drop_duplicates()` còn **1.337 dòng**.
- Chia Train/Test 80/20, `random_state=42`: **1.069 / 268** dòng.
- `OneHotEncoder(drop="first")` cho `sex`, `smoker`, `region`, ra **8 đặc trưng**. `StandardScaler` fit trên Train.
- Chọn tham số (alpha, `l1_ratio`) chỉ dùng **5-fold Cross-Validation trên Train**, với Pipeline để mã hóa/chuẩn hóa được fit lại trong từng fold. **Tập Test chỉ dùng để đánh giá cuối cùng.**

## Kết quả chính

Trên tập Test (268 dòng):

| Model | MSE (USD²) | RMSE (USD) | MAE (USD) | R² | Zero weights |
|---|---:|---:|---:|---:|---:|
| Linear Regression (OLS) | 35,478,020.68 | 5,956.34 | 4,177.05 | 0.8069 | 0 |
| Ridge (L2, α = 10) | 35,828,958.09 | 5,985.73 | 4,202.58 | 0.8050 | 0 |
| Lasso (L1, α = 100) | 36,319,178.48 | 6,026.54 | 4,202.30 | 0.8024 | 2 |
| ElasticNet (α = 100, l1_ratio = 0.9998) | 37,130,891.34 | 6,093.51 | 4,261.29 | 0.7979 | 2 |

5-fold Cross-Validation trên Train (SD là độ lệch chuẩn giữa 5 fold):

| Model | CV RMSE mean | SD RMSE | CV MAE mean | CV R² mean | SD R² |
|---|---:|---:|---:|---:|---:|
| Linear Regression (OLS) | 6,123.65 | 211.65 | 4,221.96 | 0.7228 | 0.0316 |
| Ridge (L2, α = 10) | 6,124.56 | 210.10 | 4,234.60 | 0.7228 | 0.0306 |
| Lasso (L1, α = 100) | 6,121.25 | 210.01 | 4,209.91 | 0.7231 | 0.0301 |
| ElasticNet (α = 100, l1_ratio = 0.9998) | 6,128.94 | 210.78 | 4,238.23 | 0.7226 | 0.0285 |

## Phổ Quang

- **Không có mô hình nào chính xác hơn rõ rệt.** Trên Test, OLS có sai số thấp nhất; trên CV, chênh giữa 4 mô hình khoảng 7.7 USD, nhỏ hơn rất nhiều so với SD giữa các fold (khoảng 210 USD).
- **Giá trị của Lasso/ElasticNet là mô hình gọn hơn:** cùng đưa `sex_male` và `region_northwest` về đúng 0, R² Test giảm khoảng 0,005 (Lasso) và 0,009 (ElasticNet).
- `l1_ratio = 0.9998` làm ElasticNet **gần như Lasso**, không phải cấu hình cân bằng L1/L2.
- `alpha = 100` là cấu hình nhóm chọn để minh họa chọn lọc đặc trưng, **không phải alpha tối ưu tuyệt đối**. Trong lưới thử, CV RMSE thấp nhất ở alpha ≈ 89 (6,121.19 USD); alpha 100 là 6,121.25 USD, cao hơn 0.06 USD.
- R² **không phải accuracy**. R² Test (0,798 đến 0,807) cao hơn R² CV (khoảng 0,723), nên đừng xem con số Test là độ chính xác chắc chắn.
- Trọng số mô hình là mối liên hệ thống kê, **không phải quan hệ nhân quả**.
- Thí nghiệm ElasticNet trên nhóm biến tương quan (mục 4.4.4) là **dữ liệu mô phỏng**, không phải kết quả từ `insurance.csv`.

![So sánh trọng số 4 mô hình](docs/images/5D_weights_comparison.png)

![Lasso: chọn lọc đặc trưng](docs/images/3_lasso_l1_feature_selection.png)

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/methodology.md](docs/methodology.md) | Dữ liệu, tiền xử lý, hàm mục tiêu, cách chọn tham số |
| [docs/results.md](docs/results.md) | Toàn bộ bảng số liệu và 14 biểu đồ |
| [docs/presentation_guide.md](docs/presentation_guide.md) | Phân công, mạch thuyết trình, |
| [docs/project_review.md](docs/project_review.md) | Rà soát, thay đổi, điểm chưa xác minh |
