import cv2
import numpy as np
import mediapipe as mp
import json
import logging
import time
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

frames_data = []
recording = False
frame_count = 0
current_label = "A"

# Thiết lập camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def create_label_folder(label):
    folder = f"Data/label_{label}"
    if not os.path.exists(folder):
        os.makedirs(folder)
        logger.info(f"Created folder: {folder}")
    return folder

def enhance_image(image):
    # Chuyển sang LAB để tăng độ tương phản
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Áp dụng CLAHE cho kênh L (độ sáng)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Kết hợp lại và chuyển về BGR
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Giảm nhiễu
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced

def process_frame(image, hands):
    enhanced = enhance_image(image)
    
    # Chuyển sang RGB để xử lý với MediaPipe
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    
    # Chạy mô hình hands trên toàn bộ khung hình
    results = hands.process(rgb)
    rgb.flags.writeable = True
    
    frame_out = image.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ landmarks lên hình ảnh gốc
            mp_drawing.draw_landmarks(
                frame_out,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return results, frame_out

def record_hands(hands):
    global frames_data, frame_count, recording, current_label
    folder = create_label_folder(current_label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{folder}/{current_label}_{timestamp}.json"

    frames_data = []
    frame_count = 0
    start = time.time()
    end = start + 2  # Ghi trong 2 giây

    while time.time() < end and recording:
        ret, img = cap.read()
        if not ret:
            continue
        
        results, out_img = process_frame(img, hands)

        frame_info = {
            "frame_id": frame_count,
            "timestamp": time.time() - start,
            "left_hand": None,
            "right_hand": None
        }
        
        default_landmarks = [{"id": i, "x": -1, "y": -1, "z": -1} for i in range(21)]
        left_hand_data = default_landmarks.copy()
        right_hand_data = default_landmarks.copy()

        # Xử lý landmarks nếu phát hiện tay
        if results.multi_hand_landmarks:
            for hand_idx in range(len(results.multi_hand_landmarks)):
                handedness = "Unknown"
                if results.multi_handedness and hand_idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                hand_landmarks = results.multi_hand_landmarks[hand_idx]
                
                landmarks = [
                    {"id": i, "x": lm.x, "y": lm.y, "z": lm.z}
                    for i, lm in enumerate(hand_landmarks.landmark)
                ]
                
                if handedness == "Left":
                    left_hand_data = landmarks
                elif handedness == "Right":
                    right_hand_data = landmarks
                else:
                    # Phân loại dựa trên vị trí nếu không có handedness
                    avg_x = sum(lm["x"] for lm in landmarks) / 21
                    if avg_x < 0.5:
                        left_hand_data = landmarks
                    else:
                        right_hand_data = landmarks

        frame_info["left_hand"] = left_hand_data
        frame_info["right_hand"] = right_hand_data
        frames_data.append(frame_info)

        # Hiển thị thông tin ghi hình
        cv2.putText(out_img, f"Recording: {current_label}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        remaining_time = max(0, end - time.time())
        cv2.putText(out_img, f"Time: {remaining_time:.1f}s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Hiển thị ảnh đã flip để dễ sử dụng
        flipped = cv2.flip(out_img, 1)
        cv2.imshow('Recorder', flipped)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
            break
        frame_count += 1

    recording = False
    with open(filepath, 'w') as f:
        json.dump(frames_data, f, indent=2)
        logger.info(f"Saved {len(frames_data)} frames to {filepath}")

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.7,  # Tăng độ tin cậy phát hiện
    min_tracking_confidence=0.7     # Tăng độ tin cậy theo dõi
) as hands:
    print("Chương trình ghi cử chỉ tay - Phiên bản toàn màn hình")
    print("Các lệnh:")
    print("  A, B, C, X, T: Chọn nhãn (A, B, C, Xin_chao, Cam_on)")
    print("  S: Bắt đầu ghi hình (2 giây)")
    print("  Q: Thoát chương trình")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        _, out_frame = process_frame(frame, hands)
        cv2.putText(out_frame, f"Label: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị hướng dẫn sử dụng
        cv2.putText(out_frame, "A/B/C/X/T: Change label", (10, frame.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out_frame, "S: Start recording", (10, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out_frame, "Q: Quit", (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Hiển thị ảnh đã flip để dễ sử dụng
        flipped = cv2.flip(out_frame, 1)
        cv2.imshow('Recorder', flipped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            current_label = "A"
        elif key == ord('b'):
            current_label = "B"
        elif key == ord('c'):
            current_label = "C"
        elif key == ord('x'):
            current_label = "Xin_chao"
        elif key == ord('t'):
            current_label = "Cam_on"
        elif key == ord('e'):
            current_label = ""
        elif key == ord('s'):
            recording = True
            record_hands(hands)
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
