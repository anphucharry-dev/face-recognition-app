import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# --- 1. CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(page_title="App Nhận Diện Khuôn Mặt", page_icon="👤", layout="centered")
st.title("🔥 Ứng Dụng Nhận Diện Khuôn Mặt Lớp Học")
st.markdown("Hệ thống AI tự động dò mặt, làm nét và dự đoán danh tính bằng MobileNetV2.")

# --- 2. TẢI MÔ HÌNH VÀ CẤU HÌNH ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mobilenet_face_model.keras')

model = load_model()

# CHÚ Ý: Nhớ điền đủ tên 22 thành viên vào đây giống y hệt lúc train
class_names = sorted([
    'Nguyen_Van_A', 'Tran_Thi_B', 'Doan_Hung', 'Le_Tuan_Thanh',
])
num_classes = len(class_names)
IMG_HEIGHT, IMG_WIDTH = 200, 200

# --- 3. CÁC HÀM TIỀN XỬ LÝ (SỬ DỤNG OPENCV SIÊU NHẸ) ---
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def process_and_predict(image_rgb):
    # Khởi tạo bộ dò mặt của OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Dò mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, "⚠️ Không tìm thấy khuôn mặt nào! Vui lòng thử lại."

    # Cắt khuôn mặt đầu tiên tìm thấy kèm thêm lề 15%
    x, y, w, h = faces[0]
    pad_y = int(h * 0.15)
    pad_x = int(w * 0.15)
    
    img_h, img_w = image_rgb.shape[0], image_rgb.shape[1]
    top_pad = max(0, y - pad_y)
    bottom_pad = min(img_h, y + h + pad_y)
    left_pad = max(0, x - pad_x)
    right_pad = min(img_w, x + w + pad_x)

    face_crop = image_rgb[top_pad:bottom_pad, left_pad:right_pad]
    face_clahe = apply_clahe(face_crop)

    # Ép chuẩn 200x200
    face_tensor = tf.convert_to_tensor(face_clahe)
    img_resized = tf.image.resize(face_tensor, [IMG_HEIGHT, IMG_WIDTH])
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    # AI Dự đoán
    predictions = model.predict(img_array)
    score = predictions[0]
    
    index_du_doan = np.argmax(score)
    ten_du_doan = class_names[index_du_doan]
    do_tu_tin = score[index_du_doan] * 100
    
    return face_clahe, (ten_du_doan, do_tu_tin)

# --- 4. GIAO DIỆN NGƯỜI DÙNG ---
tab1, tab2 = st.tabs(["📸 Mở Webcam", "📂 Tải ảnh lên"])
img_file_buffer = None

with tab1:
    st.info("Hãy cho phép trình duyệt sử dụng Camera. Đảm bảo ánh sáng chiếu rõ khuôn mặt.")
    cam_buffer = st.camera_input("Chụp ảnh bằng Webcam")
    if cam_buffer:
        img_file_buffer = cam_buffer

with tab2:
    upload_buffer = st.file_uploader("Chọn file ảnh (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])
    if upload_buffer:
        img_file_buffer = upload_buffer

# --- 5. XỬ LÝ KẾT QUẢ HIỂN THỊ ---
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image_rgb = np.array(image)

    # Chuyển ảnh RGBA (nếu có) về RGB
    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[..., :3]

    st.markdown("---")
    st.subheader("⚙️ Kết quả phân tích AI")
    
    with st.spinner('Đang dò tìm và phân tích khuôn mặt...'):
        face_crop, result = process_and_predict(image_rgb)
        
        if face_crop is None:
            st.error(result)
        else:
            ten, do_tu_tin = result
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_rgb, caption="Ảnh Gốc", use_container_width=True)
            with col2:
                st.image(face_crop, caption="Khuôn mặt AI trích xuất", use_container_width=True)
                
                if do_tu_tin > 70:
                    st.success(f"**Danh tính:** {ten.replace('_', ' ').upper()} \n\n **Độ tự tin:** {do_tu_tin:.2f}%")
                else:
                    st.warning(f"**Danh tính (Thiếu chắc chắn):** {ten.replace('_', ' ').upper()} \n\n **Độ tự tin:** {do_tu_tin:.2f}%")
