import cv2
import numpy as np
import time

# Open webcam
cap = cv2.VideoCapture(0)
time.sleep(3)   # give camera time to adjust

# Capture the background (30 frames)
for i in range(30):
    ret, background = cap.read()
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = np.flip(frame, axis=1)   # flip for natural viewing
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 🎯 Blue cloak HSV range
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    cloak_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean mask
    cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
    inverse_mask = cv2.bitwise_not(cloak_mask)

    # Replace cloak region with background
    res1 = cv2.bitwise_and(background, background, mask=cloak_mask)
    res2 = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Show output
    cv2.imshow("Harry Potter's Invisible Cloak (Blue)", final_output)

    # Quit with 'q'
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
