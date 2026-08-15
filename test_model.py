import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Nạp lại mô hình đã lưu
print("Đang nạp mô hình...")
model = load_model('emotion_model.h5')

# Định nghĩa danh sách các cảm xúc (theo đúng thứ tự của FER2013)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


image_path = 'khuon_mat.png'
# Lưu ý: OpenCV đọc ảnh dạng BGR, nhưng FER2013 là Grayscale
image = cv2.imread(image_path)

if image is None:
    print("Không tìm thấy ảnh tại đường dẫn đã cho!")
    exit()

# TIỀN XỬ LÝ ẢNH
# Chuyển sang ảnh xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize về kích thước
resized_image = cv2.resize(gray_image, (48, 48))

# BChuẩn hóa dữ liệu (0 -> 1)
normalized_image = resized_image / 255.0

#
# Từ (48, 48) thành (1, 48, 48, 1)
input_image = np.expand_dims(normalized_image, axis=0)
input_image = np.expand_dims(input_image, axis=-1)

# 4. ĐƯA VÀO MÔ HÌNH
print("Đang dự đoán cảm xúc...")
predictions = model.predict(input_image)

# Lấy chỉ số có xác suất cao nhất (argmax)
max_index = np.argmax(predictions[0])
predicted_emotion = emotion_labels[max_index]
confidence = predictions[0][max_index] # Độ tự tin

print(f"-> Cảm xúc dự đoán: {predicted_emotion} ({confidence*100:.2f}%)")