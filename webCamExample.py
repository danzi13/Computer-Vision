import cv2
import numpy as np

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("s - Snapshot")
print("h - Show Histograms")
print("esc - Quit program")

while True:
    # Capture frame-by-frame from webcam
    retval, img = cap.read()
    res_scale = 0.5  # Rescale the input image
    img = cv2.resize(img, (0, 0), fx=res_scale, fy=res_scale)

    lower = np.array([0,130,60])
    upper = np.array([100,200, 140])
    objmask = cv2.inRange(img, lower,upper)

    kernel = np.ones((5,5), np.uint8)
    objmask = cv2.morphologyEx(objmask, cv2.MORPH_CLOSE, kernel=kernel)
    objmask = cv2.morphologyEx(objmask, cv2.MORPH_DILATE, kernel=kernel)

    cv2.imshow("image after morph", objmask)
    # Wait for a key press
    action = cv2.waitKey(1)


    if action == 27:  # esc key to quit
        break
    elif action == ord('s'):  # capture and annotate
        cap_img = img.copy()  # copy the current frame

        # Add text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cap_img, 'I can write or draw on an image!!', (10, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw a red line on the image
        cv2.line(cap_img, (100, 100), (300, 300), (0, 0, 255), 4)

        # Show the captured image with annotations
        cv2.imshow("Captured Image", cap_img)


# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
