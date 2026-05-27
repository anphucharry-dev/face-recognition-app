import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# --- 1. CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(page_title="App Nhận Diện Khuôn Mặt", page_icon="👤", layout="centered")
st.title("🔥 Ứng Dụng Nhận Diện Khuôn Mặt Lớp Học")
st.markdown("Hệ thống AI tự động dò mặt, làm nét và dự đoán danh tính bằng MobileNetV2.")

# --- 2. TẢI MÔ HÌNH VÀ CẤU HÌNH (DÙNG CACHE ĐỂ TỐI ƯU TỐC ĐỘ) ---
@st.cache_resource
def load_model():
    # LƯU Ý: Đảm bảo file .keras nằm cùng thư mục với file app.py này
    # Hoặc thay đổi đường dẫn tuyệt đối tới file mô hình của bạn
    return tf.keras.models.load_model('mobilenet_face_model.keras')

model = load_model()

# Cập nhật danh sách 22 tên thành viên trong lớp bạn
class_names = sorted([
    'Nguyen_Van_A', 'Tran_Thi_B', 'Doan_Hung', 'Le_Tuan_Thanh',
    # ... (Thêm cho đủ 22 người)
])
num_classes = len(class_names)
IMG_HEIGHT, IMG_WIDTH = 200, 200

# --- 3. CÁC HÀM TIỀN XỬ LÝ (GIỮ NGUYÊN TỪ BẢN TRƯỚC) ---
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def process_and_predict(image_rgb):
    # Khởi tạo bộ dò mặt siêu nhẹ của OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # OpenCV cần ảnh xám để dò mặt nhanh hơn
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Dò mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, "⚠️ Không tìm thấy khuôn mặt nào! Vui lòng thử lại."

    # Lấy khuôn mặt đầu tiên (x, y là tọa độ góc trái trên, w, h là chiều rộng, cao)
    x, y, w, h = faces[0]
    
    # Thêm lề (margin) 15% để cắt không bị quá sát
    pad_y = int(h * 0.15)
    pad_x = int(w * 0.15)
    
    img_h, img_w = image_rgb.shape[0], image_rgb.shape[1]
    
    top_pad = max(0, y - pad_y)
    bottom_pad = min(img_h, y + h + pad_y)
    left_pad = max(0, x - pad_x)
    right_pad = min(img_w, x + w + pad_x)

    # Cắt khuôn mặt
    face_crop = image_rgb[top_pad:bottom_pad, left_pad:right_pad]
    
    # Cân bằng sáng CLAHE
    face_clahe = apply_clahe(face_crop)

    # Ép kích thước chuẩn 200x200 cho MobileNetV2
    face_tensor = tf.convert_to_tensor(face_clahe)
    img_resized = tf.image.resize(face_tensor, [IMG_HEIGHT, IMG_WIDTH])
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    # Dự đoán
    predictions = model.predict(img_array)
    score = predictions[0]
    
    index_du_doan = np.argmax(score)
    ten_du_doan = class_names[index_du_doan]
    do_tu_tin = score[index_du_doan] * 100
    
    return face_clahe, (ten_du_doan, do_tu_tin)
