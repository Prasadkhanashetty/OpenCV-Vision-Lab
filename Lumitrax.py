import cv2
import numpy as np
import os
from datetime import datetime


SAVE_FOLDER = "captures"
os.makedirs(SAVE_FOLDER, exist_ok=True)

CAMERA_INDEX = 0   # webcam


def lumitrax_style(img):
    """
    Highlights scratches / edges / misalignment
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Blur for noise removal
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 60, 150)

    # Morphology to strengthen lines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Invert for LumiTrax dark style
    result = 255 - edges

    # Convert to BGR for display
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result


# Main Camera Loop

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Camera not found!")
    exit()

print("Press P = capture image")
print("Press Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, "P = Capture | Q = Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow("Live Camera", display)

    key = cv2.waitKey(1) & 0xFF

    # Press P to capture
    if key == ord('p'):
        original = frame.copy()

        # Process image
        processed = lumitrax_style(original)

        # Combine side by side
        combined = np.hstack((original, processed))

        # Save image
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        path = os.path.join(SAVE_FOLDER, filename)
        cv2.imwrite(path, combined)

        cv2.imshow("Captured Result", combined)
        print(f"Saved: {path}")

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
