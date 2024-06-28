import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture.
cap = cv2.VideoCapture(0)

# Read a frame to get the dimensions.
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read a frame from the webcam")

# Create a blank image for drawing the trajectory with the same size as the frame.
trajectory = np.zeros_like(frame)

# Store the previous center point.
prev_center = None

while True:
    # Capture frame-by-frame.
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Flip the frame horizontally for a mirror effect.
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands.
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the wrist (landmark 0).
            wrist = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            center = (int(wrist.x * w), int(wrist.y * h))

            # Draw the hand landmarks.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # If there's a previous center point, draw a line from it to the current center point.
            if prev_center is not None:
                cv2.line(trajectory, prev_center, center, (0, 255, 0), 2)

            # Update the previous center point.
            prev_center = center

    # Combine the trajectory with the current frame.
    combined = cv2.addWeighted(frame, 1, trajectory, 1, 0)

    # Display the resulting frame.
    cv2.imshow('Hand Tracking', combined)

    # Check for key presses.
    key = cv2.waitKey(1) & 0xFF

    # Break the loop on 'q' key press.
    if key == ord('q'):
        break
    # Clear the trajectory on 'd' key press.
    elif key == ord('d'):
        trajectory = np.zeros_like(frame)
        prev_center = None

# Release the capture and destroy all windows.
cap.release()
cv2.destroyAllWindows()
