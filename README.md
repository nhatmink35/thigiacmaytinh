Dựa trên nội dung file notebook `ucf-crime-convlstm-take-2 (1).ipynb` mà bạn cung cấp, tôi đã phân tích và tổng hợp lại thành một file `README.md` chuyên nghiệp. File này bao gồm mô tả dự án, kiến trúc mô hình, quy trình xử lý dữ liệu và cách hệ thống dự đoán hoạt động.

Dưới đây là nội dung README đề xuất:

-----

# 📹 CCTV Anomaly Detection System (UCF-Crime Dataset)

Dự án này xây dựng một hệ thống Deep Learning để phát hiện và phân loại các hành vi bất thường (bạo lực, trộm cắp, tai nạn...) trong video giám sát (CCTV). Hệ thống sử dụng kiến trúc lai **CNN-RNN** kết hợp với cơ chế dự đoán thống nhất (Unified Prediction System) để tăng độ chính xác.

## 🚀 Tính Năng Nổi Bật

  * **Kiến trúc hiệu quả:** Sử dụng **MobileNetV2** (Backbone nhẹ) kết hợp với **LSTM** để xử lý chuỗi thời gian.
  * **Transfer Learning & Fine-tuning:** Tận dụng trọng số ImageNet và fine-tune các block cuối của MobileNetV2.
  * **Data Pipeline tối ưu:** Sử dụng `tf.data` với augmentation mô phỏng nhiễu CCTV (Gaussian noise, JPEG quality compression).
  * **Hệ thống dự đoán thông minh (3 lớp):**
    1.  **Temporal Analysis:** Phân tích trượt (sliding window) để tìm khoảnh khắc bất thường.
    2.  **Ensemble Voting:** Tổng hợp kết quả từ nhiều đoạn clip.
    3.  **Class Grouping:** Nhóm các hành vi tương tự (ví dụ: Robbery + Stealing -\> Theft) để giảm tỷ lệ dương tính giả.

## 🧠 Kiến Trúc Mô Hình (Hybrid CNN-RNN)

Mô hình nhận đầu vào là một chuỗi khung hình `(Batch, Sequence_Length, Height, Width, 3)`.

1.  **TimeDistributed Layer:** Áp dụng **MobileNetV2** (đã bỏ lớp Top, Pooling=Avg) lên từng khung hình trong chuỗi để trích xuất đặc trưng.
      * *Input:* `(32, 160, 160, 3)`
      * *Output:* `(32, 1280)` (Vector đặc trưng)
2.  **LSTM Layer:** Học mối quan hệ thời gian giữa các khung hình.
      * *Units:* 128
3.  **Classifier Head:**
      * Dropout -\> Dense (128) -\> Dropout -\> Dense (14, Softmax).

## 📂 Dữ Liệu (UCF-Crime Dataset)

Hệ thống được huấn luyện trên 14 lớp hành vi:

  * **Bình thường:** `NormalVideos`
  * **Bất thường:** `Abuse`, `Arrest`, `Arson` (Phóng hỏa), `Assault` (Tấn công), `Burglary` (Đột nhập), `Explosion`, `Fighting`, `RoadAccidents`, `Robbery`, `Shooting`, `Shoplifting`, `Stealing`, `Vandalism`.

## ⚙️ Cấu Hình & Hyperparameters

  * **Image Size:** 160x160
  * **Sequence Length:** 32 frames (mỗi clip)
  * **Batch Size:** 8 (Tối ưu cho GPU T4/P100)
  * **Optimizer:** Adam (Learning Rate = 1e-5 cho fine-tuning)
  * **Loss Function:** Sparse Categorical Crossentropy
  * **Class Weighting:** Sử dụng chiến lược `balanced` để xử lý mất cân bằng dữ liệu.

## 🛠️ Quy Trình Dự Đoán (Inference Pipeline)

Hệ thống sử dụng hàm `predict_video_unified` để phân tích video đầu vào:

1.  **Cắt Video:** Chia video dài thành các clip 32 frames chồng lấn nhau (Overlap 75%).
2.  **Lọc Bất Thường:** Loại bỏ các clip có xác suất `NormalVideos` cao (ngưỡng anomaly \> 12%).
3.  **Voting:** Các clip còn lại sẽ "bỏ phiếu" cho lớp hành vi mà nó dự đoán.
4.  **Grouping Logic:** Nếu số phiếu bị phân tán (ví dụ: một ít cho `Robbery`, một ít cho `Stealing`), hệ thống sẽ gom nhóm thành `Theft` để đưa ra quyết định chắc chắn hơn.
      * *Các nhóm:* Theft, Violence, Property, Traffic, Weapon.

## 📊 Kết Quả Thực Nghiệm

Dựa trên log huấn luyện trong notebook:

  * **Validation Accuracy:** \~97.32%
  * **Validation Loss:** \~0.0898

## 📦 Yêu Cầu Cài Đặt

```python
import tensorflow as tf
import cv2
import numpy as np
import sklearn
```

## 📝 Hướng Dẫn Sử Dụng (Kaggle Notebook)

1.  **Training:**

      * Set biến `train_model = True`.
      * Hệ thống sẽ tự động tạo pipeline dữ liệu và huấn luyện mô hình.
      * Mô hình tốt nhất được lưu tại `best_convlstm_gap_stable.h5`.

2.  **Inference (Dự đoán):**

      * Cung cấp đường dẫn video vào thư mục test.
      * Chạy Cell 6 để xem phân tích chi tiết từng giây (Top suspicious timestamps) và kết luận cuối cùng.

-----

*Dự án được thực hiện trên môi trường Kaggle với GPU NVIDIA Tesla T4.*
