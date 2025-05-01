# hole_detector.py

import cv2
import numpy as np
from detector.arrow_detector import calculate_arrow_score  # Reuse existing scoring logic

def detect_holes(img, target_center, rings, debug=False):
    holes = []

    outer_radius = rings[-1]["radius"]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # Hough Circle Transform for hole detection
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=3, maxRadius=15)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            hole_position = (x, y)
            distance = np.linalg.norm(np.array(hole_position) - np.array(target_center))
            if distance <= outer_radius:
                score = calculate_arrow_score(hole_position, target_center, rings)
                holes.append({"position": hole_position, "score": score})

                if debug:
                    cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
                    cv2.putText(img, str(score), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if debug:
        cv2.imshow("Detected Holes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return holes
