import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import face_recognition
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
    # Dò mặt
    face_locations = face_recognition.face_locations(image_rgb)
    if len(face_locations) == 0:
        return None, "⚠️ Không tìm thấy khuôn mặt nào! Vui lòng thử lại."

    # Cắt có lề (15%)
    top, right, bottom, left = face_locations[0]
    h, w = image_rgb.shape[0], image_rgb.shape[1]
    pad = int((bottom - top) * 0.15)

    top_pad = max(0, top - pad)
    bottom_pad = min(h, bottom + pad)
    left_pad = max(0, left - pad)
    right_pad = min(w, right + pad)

    face_crop = image_rgb[top_pad:bottom_pad, left_pad:right_pad]
    
    # Cân bằng sáng CLAHE
    face_clahe = apply_clahe(face_crop)

    # Ép kích thước
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

# --- 4. GIAO DIỆN NGƯỜI DÙNG ---
# Tạo 2 tab: Một cho Upload, Một cho Webcam (Tính năng rất hay của Streamlit)
tab1, tab2 = st.tabs(["📸 Mở Webcam", "📂 Tải ảnh lên"])

# Biến lưu trữ ảnh đầu vào
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

# Xử lý khi có ảnh đầu vào
if img_file_buffer is not None:
    # Đọc ảnh bằng PIL và chuyển sang Numpy array (RGB chuẩn)
    image = Image.open(img_file_buffer)
    image_rgb = np.array(image)

    # Nếu ảnh có kênh Alpha (RGBA từ png), chuyển về RGB
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
            
            # Chia cột hiển thị cho đẹp
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_rgb, caption="Ảnh Gốc", use_container_width=True)
            with col2:
                st.image(face_crop, caption="Khuôn mặt AI trích xuất & làm nét", use_container_width=True)
                
                # Hiển thị box kết quả nổi bật
                if do_tu_tin > 70:
                    st.success(f"**Danh tính:** {ten.replace('_', ' ').upper()} \n\n **Độ tự tin:** {do_tu_tin:.2f}%")
                else:
                    st.warning(f"**Danh tính (Thiếu chắc chắn):** {ten.replace('_', ' ').upper()} \n\n **Độ tự tin:** {do_tu_tin:.2f}%")