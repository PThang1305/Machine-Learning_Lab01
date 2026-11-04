# Rà soát dự án

Tài liệu này tách rõ: **nguồn đã dùng**, **kết quả chạy thực tế**, **thay đổi đã làm**, **sai khác cần biết** và **điểm chưa xác minh**.

## 1. Nguồn đã dùng

ZIP nhận được gồm `Main.ipynb`, `insurance.csv`, `Nhom9_plots.zip` (14 hình của Notebook nguồn). **Không có** bản `Nhom9_LinearModels_GitHub_Ready.zip` đã cập nhật như tài liệu bàn giao nhắc tới. Vì vậy repository này được **dựng lại từ `Main.ipynb`** và áp dụng các cải thiện mô tả trong tài liệu bàn giao (mục 4 của tài liệu đó). `data/insurance.csv` giống hệt file gốc (cùng MD5).

Không dùng `Claude ML.ipynb`, ZIP Nhóm 9 bản cũ hay số liệu của phiên bản cũ.

## 2. Thay đổi so với `Main.ipynb`

| Nhóm | Thay đổi | Lý do |
|---|---|---|
| Cấu trúc | Notebook đổi tên `Nhom9_LinearModels_Lab.ipynb`, đặt trong `notebooks/`; hình vào `docs/images/`; CSV vào `outputs/` | Cấu trúc repo rõ ràng |
| Đường dẫn | Tự tìm `data/insurance.csv` khi chạy từ thư mục gốc hoặc `notebooks/` | Không phải sửa đường dẫn khi chạy |
| Cảnh báo | Bỏ `warnings.filterwarnings("ignore", category=UserWarning)` | Không che cảnh báo; sau khi bỏ, lần chạy không phát sinh warning nào |
| Cross-Validation | Thêm mục 2.5: 5-fold CV trên Train, tiền xử lý trong `Pipeline` | Mỗi fold tự fit encoder/scaler, tránh rò rỉ |
| Chọn tham số | Ridge alpha, Lasso alpha, ElasticNet `l1_ratio` đều dùng CV trên Train; thay `LassoCV` chạy trên ma trận đã chuẩn hóa sẵn | Test chỉ dành cho đánh giá cuối cùng |
| Overfitting | Viết lại thận trọng (mục 2.4) | Một lần chia chưa đủ để khẳng định |
| Kết luận | Viết lại mục 5.2 theo CV | Chênh giữa các mô hình nhỏ hơn SD giữa các fold |
| Lịch sử | Bỏ các câu nhắc "bản cũ", "lỗi của ô ElasticNet cũ" | Repo chỉ mô tả trạng thái hiện tại |
| Tài liệu | Thêm README, 4 file `docs/*.md`, `requirements.txt`, `.gitignore` | Bàn giao, nộp bài, thuyết trình |

Tham số nhóm **không đổi**: `Ridge(alpha=10.0)`, `Lasso(alpha=100.0, max_iter=20000)`, `ElasticNet(alpha=100.0, l1_ratio=0.9998, max_iter=20000)`.

## 3. Phát hiện khi rà soát Notebook nguồn

1. **Nhận định sai về tập Test.** Notebook nguồn giải thích R² Test cao hơn Train là do "tập test tình cờ ít điểm ngoại lai chi phí cực cao hơn". Số liệu không ủng hộ: tỉ lệ dòng có `charges` > 30.000 là 11,2% ở Train và **15,7% ở Test**; `charges` trung bình 13.030 (Train) và 14.272 (Test). Câu này đã bỏ; Notebook chỉ nói điểm Test có thể là kết quả thuận lợi của lần chia này và chưa nêu nguyên nhân.
2. **Khẳng định "không có overfitting"** dựa trên một lần chia. Đã viết thận trọng hơn. R² CV trên Train (≈0,723) thấp hơn R² Test (0,798 đến 0,807), một dấu hiệu nữa cho thấy không nên xem số Test là chắc chắn.
3. **Alpha "tối ưu ≈ 42" và câu "tại đó chưa biến nào về 0"** đến từ `LassoCV` mặc định (fold không xáo trộn) chạy trên ma trận đã chuẩn hóa trên toàn Train. Với Pipeline và `KFold(shuffle=True, random_state=42)`, alpha có CV RMSE thấp nhất là ≈ 89 và tại đó đã loại đúng 2 biến. Hai điểm thấp nhất khác nhau nhưng RMSE gần như bằng nhau (alpha 42: 6.121,49 USD; alpha 89: 6.121,19 USD trong cách chia hiện tại), tức đường CV rất phẳng. Chưa tách riêng nguyên nhân gây khác biệt giữa hai cách chia.
4. Khảo sát Ridge alpha, ElasticNet `l1_ratio` và Hình 4B của Lasso từng vẽ R² trên **Test**; đã chuyển sang CV trên Train.

Các con số trong ô Markdown của Notebook đã được đối chiếu với kết quả chạy: bảng Test, R² Train/Test, thống kê phần dư (trung bình, skew, SD theo tứ phân vị, nhóm hút thuốc), VIF, ngưỡng loại biến của Lasso (≈119 cho `region_southwest`), điều kiện KKT (`|g| = 5,95` cho `region_northwest`), $\lambda_1 \approx 213.757$ và $\lambda_2 \approx 21{,}4$, các số CV ở mục 2.5, 4.2, 4.3, 4.4 và 5.2.

## 4. Nghiệm thu

| # | Hạng mục | Kết quả |
|---|---|---|
| 1 | Notebook và CSV từ ZIP mới nhất | `data/insurance.csv` cùng MD5 với file trong ZIP; Notebook dựng từ `Main.ipynb` trong ZIP |
| 2 | Restart Kernel and Run All | **37/37 ô code chạy, 0 lỗi, 0 warning**; chạy bằng nbclient/nbconvert, thư mục làm việc là `notebooks/` và đã chạy lại với thư mục làm việc là thư mục gốc; 7 file CSV giống hệt nhau giữa hai cách chạy; thời gian khoảng 30 giây |
| 3 | Đối chiếu Test và CV với CSV mới | Khớp bảng trong tài liệu bàn giao đến 2 chữ số thập phân, trừ điểm ở mục 5 dưới đây |
| 4 | 14 hình PNG, link ảnh Markdown | Đủ 14 PNG trong `docs/images/` và trong `outputs/Nhom9_plots.zip`; đã kiểm tra 13 link/ảnh trong README và `docs/*.md`: 0 link hỏng; `Nhom9_plots.zip` có đúng 14 tên file như `docs/images/` |
| 5 | README, methodology, results, presentation guide cùng bộ số liệu | Bảng Test/CV/trọng số trong README, results, presentation guide sinh từ CSV của lần chạy; đã kiểm tra chéo bảng với CSV |
| 6 | Đường dẫn tương đối, `requirements.txt`, `.gitignore` | Đường dẫn đều tương đối; hai file có mặt |
| 7 | ZIP repository | `Nhom9_LinearModels_GitHub_Ready.zip` |

## 5. Sai khác nhỏ cần biết

- **Alpha CV thấp nhất:** tài liệu bàn giao ghi 89,04; Notebook này ra **89,02** với lưới `np.logspace(-1, 3, 100)`. CV RMSE cùng là 6.121,19 USD. Tài liệu bàn giao không ghi lưới đã dùng nên không tái tạo chính xác 89,04; không chỉnh lưới để ép khớp.
- **Bảng mẫu của nhóm (Notebook mục 2.3):** MSE của Ridge trong bảng mẫu là 35.828.960,12, lần chạy này là 35.828.958,09 (lệch tương đối khoảng 6×10⁻⁸). Nằm trong dung sai đã đặt (10⁻⁶), do phiên bản scikit-learn và sai số hội tụ của solver. Không sửa.

## 6. Điểm chưa xác minh

- Chưa chạy trên máy của người dùng, trên Google Colab hay Windows. Chỉ chạy với Python 3.12.3, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2. Các cận dưới trong `requirements.txt` là cận tối thiểu theo tính năng được dùng (ví dụ `sparse_output` của `OneHotEncoder` cần scikit-learn từ 1.2), **chưa được thử** trên các phiên bản đó.
- Đã xem bằng mắt 4 trong 14 hình (phần dư OLS, Ridge path, Lasso Hình 1, Lasso Hình 4). Các hình còn lại do code không đổi sinh ra nhưng chưa xem lại từng hình.
- Các hình Ridge path, Lasso Hình 4 (`3_lasso_alpha_tuning.png`) và `4.4_elasticnet_l1_ratio.png` **đã thay đổi** so với `Nhom9_plots.zip` trong ZIP gốc (đổi sang CV). Nếu slide đang dùng bản cũ của ba hình này, cần chèn lại.
- Slide/báo cáo HTML của nhóm không có trong ZIP nên chưa đối chiếu. Nếu slide 12 đến 14 còn ghi "alpha tối ưu theo CV ≈ 42", cần sửa theo mục 3 ở trên.
- Câu mô tả "hai cụm" của phần dư nhóm hút thuốc trong Notebook mục 4.1.2 khớp với hình, nhưng chưa kiểm tra bằng phân tích định lượng.
