import cv2
import mediapipe as mp
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame
        results = holistic.process(frame_rgb)
        # Draw landmarks on the image
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return frame

def extract_keypoints(frame):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame
        results = holistic.process(frame_rgb)

        # Extract keypoints for pose
        pose_keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                pose_keypoints.extend([landmark.x, landmark.y])  # Use only x, y coordinates
        else:
            pose_keypoints = np.zeros(66)  # Assuming 33 landmarks with x, y coordinates

        # Extract keypoints for left hand
        left_hand_keypoints = []
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                left_hand_keypoints.extend([landmark.x, landmark.y])  # Use only x, y coordinates
        else:
            left_hand_keypoints = np.zeros(42)  # Assuming 21 landmarks with x, y coordinates

        # Extract keypoints for right hand
        right_hand_keypoints = []
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                right_hand_keypoints.extend([landmark.x, landmark.y])  # Use only x, y coordinates
        else:
            right_hand_keypoints = np.zeros(42)  # Assuming 21 landmarks with x, y coordinates

        # Concatenate keypoints for pose, left hand, and right hand
        return np.concatenate([pose_keypoints, left_hand_keypoints, right_hand_keypoints])

def normalize_keypoints(keypoints):
    scaler = MinMaxScaler()
    scaled_keypoints = scaler.fit_transform(keypoints.reshape(-1, 1)).flatten()
    return scaled_keypoints

def main(csv_filename):
    cap = cv2.VideoCapture(0)  # Open default camera

    # Open the CSV file in append mode
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame = process_frame(frame)

            cv2.imshow('Frame', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key >= ord('0') and key <= ord('9'):
                # Extract keypoints
                keypoints = extract_keypoints(frame)

                # Normalize keypoints
                normalized_keypoints = normalize_keypoints(keypoints)

                # Create keypoints array with the pressed key as the 0th value
                keypoints_with_label = [key - ord('0')] + normalized_keypoints.tolist()

                # Write keypoints with label to the CSV file
                csv_writer.writerow(keypoints_with_label)

            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    csv_filename = "keypoints_normalized7.csv"  # Name of the CSV file to save all normalized keypoints
    main(csv_filename)