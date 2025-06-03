import json
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm


MY_COLLECTED_KEYPOINTS_DIR = 'my_collected_keypoints'


MODEL_OUTPUT_DIR = 'asl_digit_keypoint_model_output_custom_data'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True) 

BATCH_SIZE = 32
EPOCHS = 100 
VALIDATION_SPLIT = 0.2

EXPECTED_FEATURE_DIM = 63 


def load_and_preprocess_my_keypoints():
    all_features = []
    all_labels = []

    print(f"DEBUG: Memuat data keypoint dari folder sendiri: {MY_COLLECTED_KEYPOINTS_DIR}...")

    json_files = [f for f in os.listdir(MY_COLLECTED_KEYPOINTS_DIR) if f.endswith('.json')]

    if not json_files:
        print(f"ERROR: Tidak ditemukan file JSON di {MY_COLLECTED_KEYPOINTS_DIR}. Silakan kumpulkan data terlebih dahulu.")
        return np.array([]), np.array([])

    for json_file in tqdm(json_files, desc="Memuat Keypoint JSON (Data Sendiri)"):
        file_path = os.path.join(MY_COLLECTED_KEYPOINTS_DIR, json_file)

        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            for entry in json_data:
                features = entry.get('features')
                label = entry.get('label')

                if features and label is not None and len(features) == EXPECTED_FEATURE_DIM:
                    all_features.append(features)
                    all_labels.append(label)

        except json.JSONDecodeError as e:
            print(f"ERROR: Gagal membaca JSON dari {json_file}. Kesalahan: {e}")
        except Exception as e:
            print(f"ERROR: Kesalahan tak terduga saat memproses {json_file}. Kesalahan: {e}")

    print(f"DEBUG: Total {len(all_features)} sampel keypoint dari data sendiri yang disiapkan.")

    if not all_features:
        return np.array([]), np.array([])

    return np.array(all_features), np.array(all_labels)


def train_model():
    X, y_labels_numeric = load_and_preprocess_my_keypoints()

    if len(X) == 0:
        print("ERROR: Tidak ada sampel data keypoint yang berhasil disiapkan untuk pelatihan. Mengakhiri.")
        return

    label_encoder = LabelEncoder()
    
    all_possible_digits = np.arange(10) # Angka 0 sampai 9
    label_encoder.fit(all_possible_digits)

    y_encoded = label_encoder.transform(y_labels_numeric)
    y_categorical = to_categorical(y_encoded, num_classes=len(all_possible_digits))

    num_classes = len(all_possible_digits) 
    input_shape = X.shape[1] 

    print(f"Total sampel yang akan dilatih: {len(X)}")
    print(f"Jumlah kelas unik (digit): {num_classes}")
    print(f"Bentuk input untuk model (fitur keypoint): {input_shape}")


    final_label_map = {
        int(idx): int(name) for idx, name in enumerate(label_encoder.classes_)
    }
    print(f"Mapping Akhir Label Model (Index Prediksi -> Digit ASL): {final_label_map}")

  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_encoded
    )
    print(f"Sampel pelatihan: {len(X_train)}")
    print(f"Sampel pengujian: {len(X_test)}")

  
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

  
    print("\nMemulai pelatihan model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )


    print("\nMengevaluasi model pada data pengujian...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Akurasi Model pada Data Pengujian: {accuracy*100:.2f}%\n")


    model_save_path = os.path.join(MODEL_OUTPUT_DIR, 'asl_digit_keypoint_model_custom_data.h5')
    model.save(model_save_path)
    print(f"Model berhasil disimpan di {model_save_path}")


    label_map_path = os.path.join(MODEL_OUTPUT_DIR, 'asl_digit_keypoint_label_mapping_custom_data.json')
    with open(label_map_path, 'w') as f:
        json.dump(final_label_map, f)
    print(f"Label mapping disimpan di {label_map_path}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_model()