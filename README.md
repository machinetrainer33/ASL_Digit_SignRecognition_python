Pengenalan Digit ASL Menggunakan Keypoint Tangan
==========================================

Proyek ini adalah sistem pengenalan digit American Sign Language (ASL) sederhana menggunakan titik kunci (keypoint) tangan yang dideteksi oleh MediaPipe. Proyek ini memungkinkan Anda untuk mengumpulkan data titik kunci tangan Anda sendiri, melatih model machine learning (jaringan saraf tiruan) pada data tersebut, dan kemudian menggunakan model tersebut untuk memprediksi digit ASL secara real-time melalui webcam.

PERSYARATAN
------------
Pastikan Anda sudah menginstal Python 3. Lalu, instal pustaka yang dibutuhkan menggunakan pip:

    pip install opencv-python mediapipe numpy pandas scikit-learn tensorflow matplotlib tqdm

STRUKTUR PROYEK
-----------------
.
├── my_collected_keypoints/
├── asl_digit_keypoint_model_output_custom_data/
├── collect_data.py
├── train_model.py
└── predict_realtime.py

- my_collected_keypoints/: Menyimpan data titik kunci tangan yang telah Anda kumpulkan.
- asl_digit_keypoint_model_output_custom_data/: Menyimpan model hasil pelatihan dan pemetaan label-nya.
- collect_data.py: Script untuk mengumpulkan data titik kunci tangan melalui webcam.
- train_model.py: Script untuk melatih model klasifikasi menggunakan data yang sudah dikumpulkan.
- predict_realtime.py: Script untuk melakukan prediksi digit ASL secara real-time.

PANDUAN PENGGUNAAN
-----------

Langkah 1: Kumpulkan Data Titik Kunci Tangan
----------------------------------
Pertama, kumpulkan data titik kunci tangan untuk masing-masing digit (0–9).

Jalankan script pengumpulan data:

    python collect_data.py

- Jendela webcam akan muncul. Posisikan tangan Anda dengan jelas di depan kamera.
- Untuk merekam titik kunci suatu digit, buat gerakan tangan yang sesuai dan tekan tombol angka (0–9) di keyboard.
- Pastikan tangan Anda terdeteksi dengan baik (akan muncul kerangka tangan di layar).
- Usahakan mengumpulkan setidaknya 20–30 sampel per digit untuk hasil terbaik. Semakin banyak variasi (posisi, pencahayaan), semakin baik modelnya.
- Setelah setiap rekaman, akan muncul konfirmasi di konsol.
- Untuk menghapus seluruh data yang sudah terkumpul (hati-hati!), tekan 'c' lalu konfirmasi dengan 'y'.
- Tekan 'q' untuk keluar. Data akan disimpan dalam file .json di folder my_collected_keypoints/.

Langkah 2: Latih Model Klasifikasi
--------------------------------------
Setelah mengumpulkan cukup data, latih model dengan menjalankan:

    python train_model.py

- Script ini akan memuat data dari my_collected_keypoints/, melakukan pra-pemrosesan, dan melatih jaringan saraf tiruan.
- Selama pelatihan, Anda dapat melihat perkembangan akurasi dan loss. Setelah selesai, akurasi pada data pengujian akan ditampilkan.
- Model terlatih (asl_digit_keypoint_model_custom_data.h5) dan file pemetaan label (asl_digit_keypoint_label_mapping_custom_data.json) akan disimpan di asl_digit_keypoint_model_output_custom_data/.
- Grafik akurasi dan loss juga akan ditampilkan—tutup jendela grafik untuk melanjutkan.

Langkah 3: Prediksi Secara Real-Time
------------------------------------
Gunakan model terlatih untuk melakukan prediksi digit ASL secara langsung:

    python predict_realtime.py

- Jendela webcam akan terbuka. Tunjukkan gestur digit ASL menggunakan tangan Anda.
- Model akan mendeteksi titik kunci tangan Anda dan mencoba memprediksi digit secara real-time serta menampilkannya di layar.
- Prediksi menggunakan metode pelunakan historis (historical smoothing) untuk hasil yang lebih stabil.
- Tekan 'q' untuk keluar dari mode prediksi.
