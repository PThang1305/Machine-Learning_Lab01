# Phương pháp

Tài liệu này mô tả đúng những gì Notebook `notebooks/Nhom9_LinearModels_Lab.ipynb` làm. Mỗi thuật ngữ khó có giải thích ngắn bên cạnh.

## 1. Dữ liệu

| Mục | Giá trị |
|---|---|
| File | `data/insurance.csv` |
| Số dòng ban đầu | 1.338 (7 cột: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`) |
| Giá trị thiếu | 0 |
| Dòng trùng | 1 (bị loại bằng `drop_duplicates()`) |
| Sau khi làm sạch | 1.337 dòng |
| Biến mục tiêu | `charges` (USD) |

## 2. Chia dữ liệu

`train_test_split(test_size=0.2, random_state=42)` cho **1.069 dòng Train** và **268 dòng Test**. `random_state` cố định để ai chạy cũng ra cùng kết quả.

**Quy tắc quan trọng:** Test chỉ dùng cho đánh giá cuối cùng và các biểu đồ chẩn đoán (phần dư, thực tế vs dự đoán). Mọi lựa chọn tham số (alpha, `l1_ratio`) dùng Cross-Validation trên Train.

## 3. Tiền xử lý

| Bước | Làm gì | Vì sao |
|---|---|---|
| One-hot, `drop="first"` | `sex`, `smoker`, `region` thành cột 0/1, bỏ 1 cột mỗi nhóm | Nếu giữ cả hai cột của một nhóm thì tổng luôn bằng 1 (đa cộng tuyến hoàn hảo, còn gọi là bẫy biến giả). Kết quả: 8 đặc trưng `age`, `bmi`, `children`, `sex_male`, `smoker_yes`, `region_northwest`, `region_southeast`, `region_southwest` |
| `StandardScaler` | Mỗi cột có trung bình 0, độ lệch chuẩn 1 | Phạt L1/L2 cộng trực tiếp lên trọng số, nên các biến phải cùng thang đo mới bị phạt công bằng |
| Fit trên Train, chỉ transform Test | Không dùng thông tin của Test | Tránh rò rỉ dữ liệu (data leakage: mô hình "nhìn trộm" đáp án) |

Trong Cross-Validation, encoder và scaler nằm **trong `Pipeline`** để mỗi fold tự fit lại trên phần train của fold.

## 4. Bốn mô hình và hàm mục tiêu thật sự của scikit-learn

| Mô hình | Cấu hình nhóm | Hàm mục tiêu (cần cực tiểu) |
|---|---|---|
| OLS | `LinearRegression()` | $\lVert y-Xw\rVert_2^2$ |
| Ridge (L2) | `Ridge(alpha=10.0)` | $\lVert y-Xw\rVert_2^2+\alpha\lVert w\rVert_2^2$ |
| Lasso (L1) | `Lasso(alpha=100.0, max_iter=20000)` | $\frac{1}{2n}\lVert y-Xw\rVert_2^2+\alpha\lVert w\rVert_1$ |
| ElasticNet | `ElasticNet(alpha=100.0, l1_ratio=0.9998, max_iter=20000)` | $\frac{1}{2n}\lVert y-Xw\rVert_2^2+\alpha\rho\lVert w\rVert_1+\frac{\alpha(1-\rho)}{2}\lVert w\rVert_2^2$ với $\rho$ = `l1_ratio` |

Điểm cần nhớ:

- Ridge **không** chia $2n$, Lasso và ElasticNet **có** chia. Vì vậy $\alpha=10$ của Ridge và $\alpha=100$ của Lasso **không so sánh trực tiếp** được về độ mạnh của phạt.
- Nếu slide viết Lasso dạng $\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1$ (không chia) thì $\lambda=2n\alpha$. Với $n=1069$, $\alpha=100$ tương đương $\lambda\approx 213.800$.
- `l1_ratio=0.9998` nghĩa là phần L2 chỉ còn khoảng 0,02% trọng số phạt, nên ElasticNet **gần như Lasso**. Không mô tả đây là cấu hình cân bằng L1/L2.
- Trọng số bằng 0: coi $|w|<10^{-4}$ là bằng 0 (`ZERO_TOL`).

## 5. Cách đánh giá

| Chỉ số | Ý nghĩa đơn giản |
|---|---|
| MSE, RMSE | Sai số bình phương trung bình và căn của nó (USD). RMSE phạt nặng các lần đoán sai lớn |
| MAE | Sai số tuyệt đối trung bình (USD): trung bình mỗi lần đoán lệch bao nhiêu |
| R² | Tỉ lệ biến thiên của `charges` mà mô hình giải thích được. **Không phải accuracy** |
| Zero weights | Số trọng số bị đưa về đúng 0 (chọn lọc đặc trưng) |

**Cross-Validation 5-fold:** `KFold(n_splits=5, shuffle=True, random_state=42)` trên Train. Báo cáo trung bình và độ lệch chuẩn giữa 5 fold (SD tính với `ddof=1`). Đây là cách kiểm tra ổn định hơn một lần chia: giống làm 5 bài kiểm tra thử thay vì một.

## 6. Chọn tham số (chỉ bằng CV trên Train)

| Khảo sát | Cách làm |
|---|---|
| Ridge alpha | 80 giá trị từ $10^{-2}$ đến $10^{4}$ (`np.logspace(-2, 4, 80)`), RMSE CV |
| Lasso alpha | 100 giá trị từ 0,1 đến 1000 (`np.logspace(-1, 3, 100)`), RMSE CV |
| ElasticNet `l1_ratio` | 9 giá trị {0,05; 0,1; 0,25; 0,5; 0,75; 0,9; 0,99; 0,9998; 1,0} với alpha = 100 |

Cấu hình nhóm (alpha = 10, 100, `l1_ratio` = 0,9998) **giữ nguyên**. Các khảo sát chỉ để biết cấu hình đó nằm ở đâu so với lưới, không dùng để đổi tham số cho đẹp bảng.

## 7. Những điều dự án không làm

- Không dùng Test để chọn tham số.
- Không gọi R² là accuracy.
- Không tuyên bố chắc chắn không overfitting chỉ dựa trên một lần chia.
- Không gọi alpha 100 là tối ưu tuyệt đối, không gọi ElasticNet `l1_ratio=0.9998` là cân bằng L1/L2.
- Không diễn giải trọng số thành quan hệ nhân quả.
- Không coi thí nghiệm mô phỏng ElasticNet là kết quả từ `insurance.csv`.
