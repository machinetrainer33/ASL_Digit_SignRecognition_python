ASL Digit Recognition Using Hand Keypoints
==========================================

This project is a simple American Sign Language (ASL) digit recognition system using hand keypoints detected with MediaPipe. It allows you to collect your own hand keypoint data, train a machine learning model (neural network) on that data, and then use the trained model for real-time ASL digit prediction via webcam.

REQUIREMENTS
------------
Make sure you have Python 3 installed. Then, install the required libraries using pip:

    pip install opencv-python mediapipe numpy pandas scikit-learn tensorflow matplotlib tqdm

PROJECT STRUCTURE
-----------------
.
├── my_collected_keypoints/
├── asl_digit_keypoint_model_output_custom_data/
├── collect_data.py
├── train_model.py
└── predict_realtime.py

- my_collected_keypoints/: Stores your collected hand keypoint data.
- asl_digit_keypoint_model_output_custom_data/: Stores the trained model and its label mapping.
- collect_data.py: Script to collect hand keypoint data via webcam.
- train_model.py: Script to train a classification model using the collected data.
- predict_realtime.py: Script to perform real-time ASL digit prediction.

USAGE GUIDE
-----------

Step 1: Collect Hand Keypoint Data
----------------------------------
First, collect hand keypoint data for each digit (0–9).

Run the data collection script:

    python collect_data.py

- A webcam window will open. Clearly position your hand in front of the camera.
- To record a keypoint for a digit, make the corresponding hand gesture and press the digit key (0–9) on your keyboard.
- Ensure your hand is properly detected (a hand skeleton will appear on the screen).
- Collect at least 20–30 samples per digit for best results. More variation (slight position, lighting changes) improves model accuracy.
- You’ll see confirmation in the console after each recording.
- To delete all collected data (use with caution!), press 'c' and confirm with 'y'.
- Press 'q' to exit. The collected data will be saved as .json files in the my_collected_keypoints/ folder.

Step 2: Train the Classification Model
--------------------------------------
Once enough data is collected, train the model:

    python train_model.py

- This script loads data from my_collected_keypoints/, preprocesses it, and trains a neural network.
- You’ll see accuracy and loss progress during training. After training, test accuracy will be displayed.
- The trained model (asl_digit_keypoint_model_custom_data.h5) and label mapping (asl_digit_keypoint_label_mapping_custom_data.json) will be saved in asl_digit_keypoint_model_output_custom_data/.
- Training/validation accuracy and loss graphs will also be shown—close them to continue.

Step 3: Perform Real-time Prediction
------------------------------------
Now use the trained model for real-time ASL digit prediction:

    python predict_realtime.py

- A webcam window will open. Show the ASL digit gesture with your hand.
- The model will detect your hand keypoints and attempt to predict the digit in real time, displaying the result on screen.
- Prediction uses historical smoothing for more stable output.
- Press 'q' to exit prediction mode.
