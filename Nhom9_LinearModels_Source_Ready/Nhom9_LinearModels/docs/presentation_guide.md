# Gợi ý mạch thuyết trình – Nhóm 9

## Mạch chung

1. **Bài toán:** dự đoán chi phí bảo hiểm `charges` từ tuổi, BMI, số con, giới tính, hút thuốc và khu vực.
2. **Dữ liệu:** 1.338 dòng gốc; bỏ 1 dòng trùng; chia 1.069 Train / 268 Test.
3. **Tiền xử lý:** mã hóa các biến phân loại thành số, chuẩn hóa đầu vào; fit bộ xử lý trên Train.
4. **Bốn mô hình:** OLS làm mốc; Ridge co hệ số; Lasso vừa co hệ số vừa có thể loại biến; ElasticNet kết hợp L1/L2.
5. **Đánh giá:** giải thích MSE/RMSE/MAE/R²; đọc bảng Test và CV.
6. **Kết luận:** OLS có sai số thấp nhất trên Test; khác biệt CV nhỏ; Lasso/ElasticNet có mô hình gọn hơn.

## Gợi ý phần Lasso

- Bắt đầu từ hàm mất mát: Lasso thêm hình phạt L1 vào sai số dự đoán.
- Giải thích alpha: alpha càng lớn thì phạt càng mạnh; một số trọng số có thể về 0.
- Dùng hình học: miền ràng buộc L1 có góc nhọn, vì thế nghiệm có thể nằm trên trục tọa độ.
- Giải thích Soft-Thresholding và Coordinate Descent: cập nhật từng trọng số và kéo những trọng số yếu về 0.
- Dùng KKT để kiểm tra điều kiện tối ưu cho các trọng số được giữ hoặc loại.
- Nói rõ alpha=100 là cấu hình minh họa chọn lọc đặc trưng; CV grid có điểm thấp nhất khoảng 89, nhưng chênh lệch dự đoán rất nhỏ.

## Các câu dễ được hỏi

**Tại sao phải chuẩn hóa?** Vì các biến có đơn vị/thang đo khác nhau; L1/L2 phạt trọng số nên cần đưa đặc trưng về thang đo phù hợp. Bộ chuẩn hóa chỉ fit trên Train.

**Tại sao OneHotEncoder có drop='first'?** Để tránh tạo một cột dư tuyến tính hoàn hảo trong mỗi nhóm biến giả.

**Ridge khác Lasso thế nào?** Ridge thu nhỏ trọng số nhưng thường không đưa chúng về đúng 0; Lasso có thể đưa một số trọng số về 0.

**Vì sao ElasticNet gần giống Lasso trong bài này?** Vì l1_ratio=0.9998 rất gần 1, nên thành phần L2 rất nhỏ. Đây là cấu hình nhóm chốt để minh họa, không phải cấu hình cân bằng L1/L2.

**Mô hình nào tốt nhất?** OLS có RMSE Test thấp nhất ở lần chia này, nhưng CV cho thấy khác biệt giữa bốn mô hình rất nhỏ so với dao động giữa folds. Cần tránh khẳng định OLS luôn tốt hơn.

**Trọng số smoker_yes lớn có nghĩa gì?** Trong mô hình và dữ liệu này, tình trạng hút thuốc có liên hệ mạnh với chi phí dự đoán; không được kết luận quan hệ nhân quả chỉ từ hệ số hồi quy.

## Demo Source Code

```bash
python -m src.pipeline
python -m src.predict --input data/samples/new_customers_example.csv --output reports/predictions_example.csv
```

Sau đó mở `notebooks/Main.ipynb` để trình bày biểu đồ và phần lý thuyết. Nhấn mạnh JSON là gói tham số suy luận dễ đọc; `predict.py` tự áp dụng preprocessor và công thức tuyến tính.
