# Hướng dẫn thuyết trình và vấn đáp

Tài liệu tách ba loại nội dung: **[Từ nhóm]** là thỏa thuận/phân công do nhóm cung cấp, **[Kết quả chạy]** là số liệu từ Notebook, **[Đề xuất]** là gợi ý cách trình bày.

## 1. Phân công **[Từ nhóm]**

| # | Thành viên | Phần |
|---|---|---|
| 1 | Nguyễn Đắc Thắng | Overfitting, đa cộng tuyến, Ridge |
| 2 | Trần Nguyễn Gia Huy | Tổng quan hồi quy, Linear Regression/OLS |
| 3 | Bì Phúc Thắng | Lasso |
| 4 | Nguyễn Tuấn Vinh | Presentation Khái Niệm |
| 5 | Nguyễn Quốc Thắng | ElasticNet |
| 6 | Trần Minh Trí | Demo Notebook |

## 2. Mạch trình bày phần Lasso **[Đề xuất]**

Notebook mục 4.3 (slide 12 đến 14 theo phân công nhóm).

1. **Vấn đề:** OLS giữ cả 8 biến, trong đó có biến gần như không đóng góp. Cần cách "tự chọn biến".
2. **Ý tưởng:** thêm phạt L1 $\alpha\sum|w_j|$. Phạt càng mạnh, càng nhiều trọng số bị ép về **đúng 0**.
3. **Vì sao ra đúng 0:** nhìn hình thoi (L1) có đỉnh nằm trên trục, còn hình tròn (L2) không có đỉnh (Hình 2). Hoặc nhìn đạo hàm: lực kéo về 0 của L1 luôn là ±1, của L2 nhỏ dần về 0 (Hình 3).
4. **Kết quả trên `insurance.csv`** **[Kết quả chạy]**: Lasso (alpha = 100) giữ 6/8 biến, loại `sex_male` và `region_northwest`; R² Test giảm từ 0.8069 xuống 0.8024.
5. **Chọn alpha:** Hình 4. Đường CV rất phẳng; alpha 100 là giá trị minh họa của nhóm, không phải tối ưu tuyệt đối.
6. **Kết luận:** lợi ích là mô hình gọn, không phải chính xác hơn.

## 3. Câu hỏi vấn đáp thường gặp

**1. Lasso khác Ridge ở đâu?**
Cả hai đều thêm phạt vào hàm mục tiêu để kìm trọng số. Ridge phạt $\alpha\sum w_j^2$ nên chỉ co hệ số lại, không về đúng 0. Lasso phạt $\alpha\sum|w_j|$ nên đưa được một số hệ số về đúng 0, tức là chọn biến.

**2. Vì sao Lasso đưa được hệ số về đúng 0?**
Mỗi bước cập nhật của Coordinate Descent dùng soft-thresholding $S(z,\alpha)=\mathrm{sign}(z)\max(|z|-\alpha,0)$: nếu $|z|\le\alpha$ thì $w_j=0$. Về hình học, vùng ràng buộc $|w_1|+|w_2|\le t$ là hình thoi có đỉnh nằm trên trục nên đường đồng mức của sai số thường chạm đúng đỉnh.

**3. Vì sao phải chuẩn hóa trước khi dùng Lasso?**
Phạt L1 cộng thẳng giá trị tuyệt đối của trọng số. Nếu biến khác thang đo (tuổi so với BMI), biến có thang nhỏ cần trọng số lớn hơn và bị phạt nặng hơn một cách không công bằng.

**4. `region_northwest` có $w_{OLS}=-168{,}94$, lớn hơn alpha = 100, sao vẫn bị loại?**
Công thức $w=S(w_{OLS},\alpha)$ chỉ đúng khi các biến trực giao. Ba biến `region` tương quan với nhau (khoảng −0,33 đến −0,35). Lasso xét tương quan với phần dư **sau khi đã tính các biến khác**; với `region_northwest` con số đó chỉ còn $|g|=5{,}95<100$ **[Kết quả chạy, Notebook mục 4.3.4]**. Đừng nói "w_Lasso = w_OLS − 100" cho mọi biến.

**5. Vì sao dùng alpha = 100? Có phải tối ưu không?**
Không. Đây là cấu hình nhóm thống nhất để minh họa chọn lọc đặc trưng. Trong lưới thử, CV RMSE thấp nhất ở alpha ≈ 89 (6,121.19 USD); alpha 100 là 6,121.25 USD, cao hơn 0.06 USD. Cả hai cùng loại đúng 2 biến, và đường CV rất phẳng nên không có "một con số đúng duy nhất".

**6. Ridge alpha = 10 và Lasso alpha = 100, cái nào phạt mạnh hơn?**
Không so sánh trực tiếp được: Ridge không chia $2n$, Lasso chia. Quy đổi Lasso về dạng không chia thì $\lambda=2n\alpha\approx 213.800$ với $n=1069$.

**7. Lasso có dự đoán tốt hơn OLS không?**
Không trên dữ liệu này. Test: R² 0.8069 (OLS) so với 0.8024 (Lasso). CV: gần như ngang nhau (RMSE 6,123.65 so với 6,121.25, chênh nhỏ hơn nhiều so với SD giữa các fold). Giá trị của Lasso là mô hình gọn hơn.

**8. R² có phải accuracy không?**
Không. R² là tỉ lệ biến thiên của `charges` mà mô hình giải thích được. Accuracy là phần trăm đoán đúng, dùng cho phân loại.

**9. Mô hình có overfitting không?**
Overfitting là train tốt nhưng dữ liệu mới kém hơn hẳn. Trên lần chia này R² Train (≈0,73) thấp hơn R² Test (0,80 đến 0,81), nên không thấy dấu hiệu rõ rệt. Nhưng đó chỉ là một lần chia, và R² CV (≈0,723) thấp hơn R² Test, nên không tuyên bố chắc chắn.

**10. ElasticNet `l1_ratio=0.9998` có cân bằng L1 và L2 không?**
Không. Phần L2 chỉ còn khoảng 0,02% trọng số phạt ($\lambda_2\approx 21{,}4$ so với $\lambda_1\approx 213.757$), nên gần như Lasso. Hạ `l1_ratio` xuống 0,99 là R² CV tụt mạnh vì alpha = 100 đã rất lớn **[Kết quả chạy]**.

**11. Pipeline trong Cross-Validation để làm gì?**
Để encoder/scaler được fit lại trong từng fold, chỉ trên phần train của fold. Nếu chuẩn hóa trước trên cả Train rồi mới chia fold, fold dùng để chấm đã "lọt" thông tin vào bước chuẩn hóa (data leakage).

**12. Vì sao không dùng Test để chọn alpha?**
Test đóng vai dữ liệu chưa từng thấy. Nếu thử nhiều alpha rồi chọn cái tốt nhất trên Test thì điểm Test không còn đáng tin.

**13. `smoker_yes` có trọng số lớn nhất, vậy hút thuốc gây ra chi phí cao hơn?**
Không kết luận được quan hệ nhân quả. Mô hình chỉ cho thấy liên hệ thống kê trong dữ liệu này.

**14. `sex_male` bị loại có nghĩa giới tính không ảnh hưởng chi phí?**
Không. Chỉ có nghĩa với alpha = 100 và các biến còn lại, biến này không đóng góp đủ để thắng phạt L1 (trọng số OLS của nó chỉ −50,75 USD trên 1 độ lệch chuẩn).

**15. Thí nghiệm ElasticNet với hai biến gần trùng nhau lấy từ `insurance.csv` à?**
Không, đó là **dữ liệu mô phỏng** (Notebook mục 4.4.4) để minh họa lý thuyết, vì các biến trong `insurance.csv` chỉ tương quan nhẹ (VIF lớn nhất 1,68).

## 4. Việc nên làm trước khi thuyết trình **[Đề xuất]**

- Chạy Restart Kernel and Run All một lần trên máy mình, đối chiếu bảng Test trong `docs/results.md`.
- Chỉ nói "gọn hơn", không nói "chính xác hơn" khi so Lasso/ElasticNet với OLS.
- Tự giải thích được câu 4, 5 và 10 bằng lời của mình, vì đây là những câu hay bị hỏi sâu.
