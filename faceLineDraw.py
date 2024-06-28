import cv2
import numpy as np

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the cascade file was loaded correctly
if face_cascade.empty():
    raise IOError("Cannot load haarcascade_frontalface_default.xml")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Check if the video capture is initialized
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Read a frame to get the dimensions
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read a frame from the webcam")

# Create a blank image for drawing the trajectory with the same size as the frame
trajectory = np.zeros_like(frame)

# Store the previous center point
prev_center = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate the center point of the face
        center = (x + w // 2, y + h // 2)

        # If there's a previous center point, draw a line from it to the current center point
        if prev_center is not None:
            cv2.line(trajectory, prev_center, center, (0, 255, 0), 2)

        # Update the previous center point
        prev_center = center

    # Combine the trajectory with the current frame
    combined = cv2.add(frame, trajectory)

    # Display the resulting frame
    cv2.imshow('Face Tracking', combined)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
