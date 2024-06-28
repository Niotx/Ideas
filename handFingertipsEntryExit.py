import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Allow up to 2 hands.
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture.
cap = cv2.VideoCapture(0)

# Get the width and height of the frame.
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read a frame from the webcam")
h, w, _ = frame.shape

# Store the previous center points and their last side (enter or exit).
prev_centers = {}
prev_sides = {}

# Initialize counters for entries and exits.
enter_count = 0
exit_count = 0

# Initial color of the vertical line (blue).
line_color = (255, 0, 0)

# Unique ID for fingertips.
next_fingertip_id = 0
fingertip_ids = {}

# Define a function to get a unique ID for each fingertip.
def get_fingertip_id(point):
    global next_fingertip_id

    # If the point is already in the dictionary, return its ID.
    for id, prev_point in fingertip_ids.items():
        if np.linalg.norm(np.array(point) - np.array(prev_point)) < 20:
            fingertip_ids[id] = point
            return id

    # Otherwise, assign a new ID.
    fingertip_ids[next_fingertip_id] = point
    next_fingertip_id += 1
    return next_fingertip_id - 1

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
            # Get the coordinates of the fingertips.
            fingertips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
            fingertip_points = [(int(pt.x * w), int(pt.y * h)) for pt in fingertips]

            # Draw the hand landmarks.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for point in fingertip_points:
                fingertip_id = get_fingertip_id(point)

                # Determine which side the current fingertip is on.
                current_side = 'enter' if point[0] < line_x else 'exit'

                # If there's a previous center point, check for line crossing and movement.
                if fingertip_id in prev_centers:
                    prev_point = prev_centers[fingertip_id]
                    prev_side = prev_sides[fingertip_id]

                    # Determine the movement direction.
                    direction = 'forward' if point[0] > prev_point[0] else 'backward'

                    # Check if the fingertip crossed the line.
                    if prev_side != current_side:
                        if current_side == 'enter':
                            enter_count += 1
                        elif current_side == 'exit':
                            exit_count += 1

                        # Change the color of the line randomly.
                        line_color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))

                    # Draw the fingertip ID and direction.
                    cv2.putText(frame, f'ID: {fingertip_id} ({direction})', (point[0], point[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Update the previous center point and side for the current fingertip.
                prev_centers[fingertip_id] = point
                prev_sides[fingertip_id] = current_side

                # Draw circles on the fingertips.
                cv2.circle(frame, point, 10, (0, 255, 0), -1)

    # Display the counters on the frame.
    cv2.putText(frame, f'Entries: {enter_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Exits: {exit_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame.
    cv2.imshow('Hand Tracking', frame)

    # Check for key presses.
    key = cv2.waitKey(1) & 0xFF

    # Break the loop on 'q' key press.
    if key == ord('q'):
        break
    # Clear the counters on 'd' key press.
    elif key == ord('d'):
        prev_centers = {}
        prev_sides = {}
        enter_count = 0
        exit_count = 0
        fingertip_ids = {}
        next_fingertip_id = 0

# Release the capture and destroy all windows.
cap.release()
cv2.destroyAllWindows()
