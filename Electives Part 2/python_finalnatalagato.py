import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# Load YOLOv8 model
model = YOLO("FingerDetect.pt")

# Initialize the video source
source = cv2.VideoCapture(0)
if not source.isOpened():
    print("Error: Unable to access the camera.")
    exit()

win_name = "Finger Detection"
cv2.namedWindow(win_name)

# Create a dedicated folder for saving fingerprints
if not os.path.exists('Fingerprints'):
    os.makedirs('Fingerprints')

scale_factor = 1.5  # Scaling factor for frames
confidence_threshold = 0.5  # Confidence threshold

while True:
    has_frame, frame = source.read()
    if not has_frame:
        print("Error: Unable to read frame from camera.")
        break

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    # YOLOv8 Inference
    results = model(small_frame)
    predictions = results[0].boxes.xyxy  # Bounding boxes (x_min, y_min, x_max, y_max)
    confidences = results[0].boxes.conf  # Confidence scores
    fingerprint_count = 0

    for i, box in enumerate(predictions):
        confidence = confidences[i]
        if confidence < confidence_threshold:
            continue

        x_min, y_min, x_max, y_max = map(int, box)
        # Expand ROI slightly
        x_min, y_min = max(x_min - 10, 0), max(y_min - 10, 0)
        x_max, y_max = min(x_max + 10, small_frame.shape[1]), min(y_max + 10, small_frame.shape[0])

        # Crop ROI
        crop = small_frame[y_min:y_max, x_min:x_max]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Enhance details using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_crop)

        # Enhance details using adaptive thresholding
        fingerprint = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Smooth the fingerprint region using Gaussian blur
        gaussian = cv2.GaussianBlur(gray_crop, (5, 5), 0)

        # Apply the Sobel operator in the X and Y directions
        sobel_x = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gaussian, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate the magnitude of the gradients
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize to 0 to 255
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))

        # Invert the edges to enhance visibility
        magnitude = cv2.bitwise_not(magnitude)

        # Use dilation to thicken the edges
        kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for dilation
        thickened_edges = cv2.dilate(magnitude, kernel, iterations=1)

        # Combine the thickened edges with the fingerprint using bitwise AND
        improved_fingerprint = cv2.bitwise_and(fingerprint, thickened_edges)

        # Highlight ROI on the original frame
        cv2.rectangle(frame, (int(x_min / scale_factor), int(y_min / scale_factor)), 
                      (int(x_max / scale_factor), int(y_max / scale_factor)), (0, 255, 0), 2)
        cv2.putText(frame, f"Finger {fingerprint_count + 1} ({confidence:.2f})", 
                    (int(x_min / scale_factor), int(y_min / scale_factor) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Automatically save the fingerprint
        cv2.putText(frame, "Press S to save", (10, frame.shape[0] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Fingerprint", fingerprint)
        cv2.imshow("Improved Fingerprint", improved_fingerprint)
        if cv2.waitKey(1) == ord('s'):
          timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
          cv2.imwrite(f"./Fingerprints/Fingerprint_{fingerprint_count + 1}_{timestamp}.png", fingerprint)
          cv2.imwrite(f"./Fingerprints/Fingerprint_Improved_Borders_{fingerprint_count + 1}_{timestamp}.png", improved_fingerprint)
          print(f"Fingerprint {fingerprint_count + 1} automatically saved in the 'Fingerprints' folder.")

        fingerprint_count += 1

    if fingerprint_count == 0:
        cv2.putText(frame, "No fingers detected! Adjust position.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the live feed
    cv2.imshow(win_name, frame)

    # Key controls
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

# Release resources
source.release()
cv2.destroyAllWindows()
