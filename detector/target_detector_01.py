import cv2
import numpy as np
from .utils import create_color_mask

def detect_3spot_faces(img, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blue_mask = create_color_mask(hsv, '#02bce3', tolerance=(8, 80, 80))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # filter small noise
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        candidates.append((int(x), int(y), int(radius)))

    if not candidates:
        return []

    # Select only the most centered candidate
    img_center = np.array([img.shape[1] // 2, img.shape[0] // 2])
    best = min(candidates, key=lambda c: np.linalg.norm(np.array([c[0], c[1]]) - img_center))

    if debug:
        debug_img = img.copy()
        cv2.circle(debug_img, (best[0], best[1]), best[2], (255, 0, 0), 2)
        cv2.circle(debug_img, (best[0], best[1]), 2, (0, 0, 255), 2)
        cv2.imshow("Detected Main Blue Ring", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return [best]
