# Báo cáo rà soát dự án sau khi tách Source Code

## Phạm vi

Bản tham chiếu được dùng để xây dựng phiên bản này là ZIP `Machine-Learning_Lab01-main.zip` do người dùng cung cấp gần nhất trong cuộc trao đổi. Notebook học thuật đã được đặt tên `notebooks/Main.ipynb` theo quyết định của nhóm; nội dung 85 ô (36 code, 49 Markdown) được giữ, các khối xử lý chính được chuyển thành lời gọi module trong `src/`.

## Các quyết định kiến trúc đã thực hiện

- Dùng scikit-learn cho bốn estimator chính: LinearRegression, Ridge, Lasso và ElasticNet.
- Tách source thành `config.py`, `preprocess.py`, `models.py`, `tune.py`, `train.py`, `evaluate.py`, `persistence.py`, `predict.py` và `pipeline.py`.
- Notebook giữ lý thuyết, công thức, phân tích sâu, biểu đồ và kết luận; các chức năng dùng chung import từ `src/`.
- Không dùng `.pkl`/`.joblib`. Thông tin suy luận được lưu bằng bốn model JSON và một `preprocessor.json`.
- Dữ liệu processed là bốn CSV Train/Test ở dạng đặc trưng gốc sau làm sạch/chia tập; mã hóa và chuẩn hóa diễn ra trong Pipeline, fit trên Train.
- `predict.py` nhận CSV và trả về một CSV gồm dự đoán của cả bốn mô hình.

## Kiểm tra đã thực hiện

1. `python -m src.pipeline`: chạy thành công; tạo các CSV Train/Test, bảng metrics/CV và năm JSON trong `models/`.
2. `python -m src.predict --input data/samples/new_customers_example.csv --output reports/predictions_example.csv`: chạy thành công và tạo đủ bốn cột dự đoán.
3. Kiểm chứng trực tiếp 25 dòng Test: dự đoán tính từ JSON khớp tuyệt đối với dự đoán estimator scikit-learn trong môi trường kiểm tra (max absolute difference = 0.0 cho cả bốn mô hình).
4. Kiểm thử unknown category: hành vi của JSON tương đương `OneHotEncoder(handle_unknown="ignore")`.
5. `python -m unittest discover -s tests -v`: 3/3 test thành công.
6. Notebook: 36/36 ô code hoàn thành; không có output error; 14 PNG được lưu trong `reports/figures/`.

## Kết quả tham chiếu

Kết quả Test trong lần chạy hiện tại khớp bảng tham chiếu của Notebook: R² xấp xỉ 0.8069 (OLS), 0.8050 (Ridge), 0.8024 (Lasso), 0.7979 (ElasticNet). Lasso và ElasticNet loại `sex_male` và `region_northwest` theo ngưỡng `1e-4`.

5-fold CV chạy trên Train; preprocessor được fit riêng trong từng fold. Lưới Lasso cho alpha tốt nhất xấp xỉ 89.02; alpha nhóm dùng là 100 để minh họa chọn lọc đặc trưng. Không coi alpha 100 là tối ưu tuyệt đối.

## Những điều cần lưu ý khi sử dụng

- JSON là gói tham số phục vụ suy luận cho mô hình tuyến tính, không phải object sklearn. Không thể gọi `pickle.load` hay `joblib.load` trên các file này.
- `predict.py` tự tái tạo One-Hot Encoding và StandardScaler từ `preprocessor.json`. Unit test xác nhận kết quả với scikit-learn trong môi trường kiểm tra.
- `data/processed/` và `models/*.json` được tạo/ghi đè khi chạy pipeline. Hãy commit các file này nếu nhóm muốn giảng viên có thể xem kết quả mà không chạy lại.
- Cần chạy lệnh từ thư mục gốc của repository để các module tìm được đường dẫn mặc định.
- Phiên bản môi trường kiểm thử hiện tại: Python 3.13.5, scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, Matplotlib 3.10.8. Các con số cuối có thể khác nhẹ ở môi trường khác.
