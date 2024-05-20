import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture.
cap = cv2.VideoCapture(0)

# Read a frame to get the dimensions.
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read a frame from the webcam")

# Get the width and height of the frame.
h, w, _ = frame.shape

# Create a blank image for drawing the trajectory with the same size as the frame.
trajectory = np.zeros_like(frame)

# Store the previous center point.
prev_center = None

# Initialize the counter for line crossings.
line_crossings = 0

# Initial color of the vertical line (blue).
line_color = (255, 0, 0)

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

    # Draw the vertical line in the center.
    line_x = w // 2
    cv2.line(frame, (line_x, 0), (line_x, h), line_color, 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the wrist (landmark 8).
            wrist = hand_landmarks.landmark[8]
            center = (int(wrist.x * w), int(wrist.y * h))
        
            # Draw the hand landmarks.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # If there's a previous center point, draw a line from it to the current center point.
            if prev_center is not None:
                #cv2.line(trajectory, prev_center, center, (0, 255, 0), 2)

                # Check if the hand crosses the vertical line.
                if (prev_center[0] < line_x and center[0] > line_x) or (prev_center[0] > line_x and center[0] < line_x):
                    line_crossings += 1
                    # Change the color of the line randomly.
                    line_color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))

            # Update the previous center point.
            prev_center = center

    # Combine the trajectory with the current frame.
    combined = cv2.addWeighted(frame, 1, trajectory, 1, 0)

    # Display the counter on the frame.
    cv2.putText(combined, f'Line Crossings: {line_crossings}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame.
    cv2.imshow('Hand Tracking', combined)

    # Check for key presses.
    key = cv2.waitKey(1) & 0xFF

    # Break the loop on 'q' key press.
    if key == ord('q'):
        break
    # Clear the trajectory and reset the counter on 'd' key press.
    elif key == ord('d'):
        trajectory = np.zeros_like(frame)
        prev_center = None
        line_crossings = 0

# Release the capture and destroy all windows.
cap.release()
cv2.destroyAllWindows()
