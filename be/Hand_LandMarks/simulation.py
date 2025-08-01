def replay_gestures_from_json(json_file):
    import cv2
    import numpy as np
    import json
    import mediapipe as mp

    mp_hands = mp.solutions.hands

    with open(json_file, 'r') as f:
        frames_data = json.load(f)

    cv2.namedWindow('Gesture Replay', cv2.WINDOW_NORMAL)

    for frame in frames_data:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"Frame: {frame['frame_id']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Time: {frame['timestamp']:.2f}s", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for hand_label, color in [("left_hand", (0, 255, 0)), ("right_hand", (0, 0, 255))]:
            landmarks = frame.get(hand_label, [])
            # Kiểm tra có đủ dữ liệu x, y không
            valid_landmarks = [lm for lm in landmarks if "x" in lm and "y" in lm and lm["x"] != -1 and lm["y"] != -1]
            if len(valid_landmarks) == 21:
                # Vẽ các điểm
                for lm in valid_landmarks:
                    x = int(lm["x"] * 640)
                    y = int(lm["y"] * 480)
                    cv2.circle(img, (x, y), 5, color, -1)
                # Vẽ connections
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < 21 and end_idx < 21:
                        start_lm = valid_landmarks[start_idx]
                        end_lm = valid_landmarks[end_idx]
                        x1, y1 = int(start_lm["x"] * 640), int(start_lm["y"] * 480)
                        x2, y2 = int(end_lm["x"] * 640), int(end_lm["y"] * 480)
                        cv2.line(img, (x1, y1), (x2, y2), color, 2)

        cv2.imshow('Gesture Replay', img)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

# Đường dẫn file nên dùng dấu gạch chéo hoặc raw string
replay_gestures_from_json(r'C:\Users\TIN\SpringBoot\VoiceToText\be\Hand_LandMarks\Data\label_C\C_20250731_011214.json')