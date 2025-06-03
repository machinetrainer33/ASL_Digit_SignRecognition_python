import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import json
import time

OUTPUT_DATA_COLLECTION_DIR = 'my_collected_keypoints'
os.makedirs(OUTPUT_DATA_COLLECTION_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def normalize_keypoints_for_collection(hand_landmarks_list):
    if not hand_landmarks_list:
        return None

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

    return scaled_keypoints

def run_data_collector():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Tidak dapat membuka kamera. Pastikan webcam terhubung.")
        return

    print(f"--- Pengumpul Data Keypoint ---")
    print(f"Data akan disimpan di folder: {OUTPUT_DATA_COLLECTION_DIR}")
    print(f"Tekan tombol angka (0-9) untuk merekam keypoint untuk angka tersebut.")
    print(f"Tekan 'q' untuk keluar.")
    print(f"Tekan 'c' untuk membersihkan folder {OUTPUT_DATA_COLLECTION_DIR} (HATI-HATI!)")

    collected_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        display_text = "Siap Rekam..."
        
        if results.multi_hand_landmarks:
            for hand_obj in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_obj, mp_hands.HAND_CONNECTIONS)
            
            normalized_kp = normalize_keypoints_for_collection(results.multi_hand_landmarks)
            
            if normalized_kp is not None:
                display_text = "Tangan Terdeteksi. Tekan Angka..."

        else:
            display_text = "Tidak Ada Tangan Terdeteksi."

        cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Pengumpul Data Keypoint", frame)

        key = cv2.waitKey(1) & 0xFF

        if key >= ord('0') and key <= ord('9'):
            if normalized_kp is not None:
                label = int(chr(key))
                collected_data.append({'features': normalized_kp, 'label': label})
                print(f"Direkam: Angka {label}. Total sampel: {len(collected_data)}")
                cv2.putText(frame, f"Direkam: Angka {label}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("Pengumpul Data Keypoint", frame)
                cv2.waitKey(500)
            else:
                print("Tidak dapat merekam: Tangan tidak terdeteksi.")
        
        elif key == ord('c'):
            print(f"PERINGATAN: Anda akan menghapus semua file di {OUTPUT_DATA_COLLECTION_DIR}. Ketik 'y' untuk konfirmasi.")
            confirm_key = cv2.waitKey(0) & 0xFF
            if confirm_key == ord('y'):
                for f in os.listdir(OUTPUT_DATA_COLLECTION_DIR):
                    os.remove(os.path.join(OUTPUT_DATA_COLLECTION_DIR, f))
                collected_data = []
                print(f"Folder {OUTPUT_DATA_COLLECTION_DIR} berhasil dibersihkan.")
            else:
                print("Penghapusan dibatalkan.")
        
        elif key == ord('q'):
            break

    if collected_data:
        file_name = f"my_digit_keypoints_{int(time.time())}.json"
        file_path = os.path.join(OUTPUT_DATA_COLLECTION_DIR, file_name)
        with open(file_path, 'w') as f:
            json.dump(collected_data, f, indent=4)
        print(f"\nData berhasil disimpan ke: {file_path}")
        print(f"Total sampel disimpan: {len(collected_data)}")
    else:
        print("\nTidak ada data yang dikumpulkan.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    run_data_collector()