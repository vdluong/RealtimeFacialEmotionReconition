import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Nạp mô hình AI
print("Đang nạp mô hình...")
model = load_model('emotion_model.h5')

# 2. Nạp file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#7 cảm xúc
emotion_labels = ['Tuc Gian', 'Chan nan', 'So Hai', 'Vui Ve', 'Buon Ba', 'Ngac Nhien', 'Binh Thuong']

# 3. Mở Webcam
cap = cv2.VideoCapture(0)
print("Đã mở Camera. Nhấn phím 'q' trên bàn phím để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi = roi_gray / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)
        max_index = int(np.argmax(prediction))
        predicted_emotion = emotion_labels[max_index]

        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Nhan dien Cam xuc Real-time', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()