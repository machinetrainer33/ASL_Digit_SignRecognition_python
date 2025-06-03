import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json
import time
from collections import Counter
import os


MODEL_DIR = 'asl_digit_keypoint_model_output_custom_data'

MODEL_PATH = os.path.join(MODEL_DIR, 'asl_digit_keypoint_model_custom_data.h5')
LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'asl_digit_keypoint_label_mapping_custom_data.json')

EXPECTED_FEATURE_DIM = 21 * 3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def normalize_keypoints_for_prediction(hand_landmarks_list):
  
    if not hand_landmarks_list:
        
        return np.array([0.0] * EXPECTED_FEATURE_DIM).reshape(1, -1)

    hand_landmarks = hand_landmarks_list[0]

    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])

    wrist_x, wrist_y, wrist_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z

    normalized_keypoints = []
    for i in range(0, len(keypoints), 3):
        normalized_keypoints.append(keypoints[i] - wrist_x)
        normalized_keypoints.append(keypoints[i+1] - wrist_y)
        normalized_keypoints.append(keypoints[i+2] - wrist_z)

    np_norm_kp = np.array(normalized_keypoints)
    max_val = np.max(np.abs(np_norm_kp))
    if max_val > 0:
        scaled_keypoints = (np_norm_kp / max_val).tolist()
    else:
        scaled_keypoints = np_norm_kp.tolist()

    return np.array(scaled_keypoints).reshape(1, -1)


def run_realtime_prediction():
    try:
        model = load_model(MODEL_PATH)
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
        print("Model dan label mapping berhasil dimuat.")
    except Exception as e:
        print(f"ERROR: Gagal memuat model atau label mapping dari path yang ditentukan: {e}")
        print("Pastikan Anda sudah melatih dan menyimpan model keypoint terlebih dahulu, dan path-nya benar.")
        print(f"Mencoba memuat dari: {MODEL_PATH} dan {LABEL_MAP_PATH}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Tidak dapat membuka kamera. Pastikan webcam terhubung dan tidak digunakan oleh aplikasi lain.")
        return

    print("Prediksi ASL Digit (Keypoint) Dimulai. Lakukan isyarat angka (0-9). Tekan 'q' untuk keluar.")

    prediction_history = []
    MAX_HISTORY_LEN = 20 
    last_displayed_sign = "Mencari Digit..."
    DISPLAY_THRESHOLD = 0.7 
    CONSISTENCY_THRESHOLD = 0.6 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera. Mengakhiri.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        current_prediction_info = None

        if results.multi_hand_landmarks:
            for hand_obj in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_obj, mp_hands.HAND_CONNECTIONS)

            input_features = normalize_keypoints_for_prediction(results.multi_hand_landmarks)

            predictions = model.predict(input_features, verbose=0)
            predicted_class_idx = np.argmax(predictions)
            prediction_confidence = predictions[0][predicted_class_idx]
            predicted_label = label_map.get(predicted_class_idx, "Unknown")

            current_prediction_info = (predicted_label, prediction_confidence)

        if current_prediction_info:
            prediction_history.append(current_prediction_info)
        else:
            prediction_history.append(("", 0.0)) 

        if len(prediction_history) > MAX_HISTORY_LEN:
            prediction_history.pop(0)

      
        smoothed_label_to_display = "Mencari Digit..."
        if prediction_history:
         
            labels_in_history = [p[0] for p in prediction_history if p[0] != ""]

            if labels_in_history:
                most_common_pred_info = Counter(labels_in_history).most_common(1)

                if most_common_pred_info:
                    most_common_label = most_common_pred_info[0][0]
                    occurrence_count = most_common_pred_info[0][1]

                 
                    avg_confidence_for_common = np.mean([p[1] for p in prediction_history if p[0] == most_common_label])

                 
                    if (occurrence_count / len(prediction_history)) >= CONSISTENCY_THRESHOLD and \
                       avg_confidence_for_common >= DISPLAY_THRESHOLD:
                        last_displayed_sign = f"{most_common_label} ({avg_confidence_for_common*100:.2f}%)"
                    else:
                        last_displayed_sign = "Mencari Digit..."
                else:
                    last_displayed_sign = "Mencari Digit..."
            else:
                last_displayed_sign = "Tidak Ada Tangan Terdeteksi" 
        else:
            last_displayed_sign = "Mencari Digit..." 
        cv2.putText(frame, f"Prediksi: {last_displayed_sign}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("ASL Digit Keypoint Predictor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    run_realtime_prediction()