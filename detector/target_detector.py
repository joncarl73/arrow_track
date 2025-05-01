import cv2
import numpy as np
from .utils import create_color_mask

def detect_3spot_faces(img, debug=False, count_center_x_as_11=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #blue_mask = create_color_mask(hsv, '#02bce3', tolerance=(8, 80, 80)) #maple 3 spot
    blue_mask = create_color_mask(hsv, '#6dc6fe', tolerance=(8, 80, 80)) #vegas 3 spot

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        candidates.append((int(x), int(y), int(radius)))

    if not candidates:
        return []

    img_center = np.array([img.shape[1] // 2, img.shape[0] // 2])
    best = min(candidates, key=lambda c: np.linalg.norm(np.array([c[0], c[1]]) - img_center))
    center_x, center_y, outer_radius = best

    scoring_rings = []

    if count_center_x_as_11:
        # Proper scaling: X ring is half the 10 ring radius
        ten_radius = outer_radius * (1 / 5)
        x_radius = ten_radius / 2
        scoring_rings.append({"score": 11, "radius": x_radius})
        scoring_rings.append({"score": 10, "radius": ten_radius})

        # Spread remaining 9–6 over the space between 10 ring and outer edge
        step = (outer_radius - ten_radius) / 4
        for i, score in enumerate([9, 8, 7, 6]):
            radius = ten_radius + step * (i + 1)
            scoring_rings.append({"score": score, "radius": radius})
    else:
        step = outer_radius / 5
        for idx, score in enumerate([10, 9, 8, 7, 6]):
            radius = step * (idx + 1)
            scoring_rings.append({"score": score, "radius": radius})

    if debug:
        debug_img = img.copy()

        # Draw outer boundary in blue
        cv2.circle(debug_img, (center_x, center_y), outer_radius, (255, 0, 0), 2)

        # Draw center point in red
        cv2.circle(debug_img, (center_x, center_y), 2, (0, 0, 255), 2)

        for ring in scoring_rings:
            draw_radius = int(ring["radius"])
            score_text = str(ring["score"])

            # Draw scoring ring in green
            cv2.circle(debug_img, (center_x, center_y), draw_radius, (0, 255, 0), 2)

            # Place labels at 4 angles: 0°, 90°, 180°, 270°
            angles_deg = [90]
            for angle_deg in angles_deg:
                angle_rad = np.deg2rad(angle_deg)
                label_x = int(center_x + (draw_radius + 10) * np.cos(angle_rad))
                label_y = int(center_y + (draw_radius + 10) * np.sin(angle_rad))

                cv2.putText(debug_img, score_text, (label_x - 10, label_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Detected Target Face with Rings", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Detected a target at ({center_x}, {center_y}) with radius {outer_radius} pixels.")
    for ring in scoring_rings:
        print(f" - Scoring Ring: Score {ring['score']} at radius {int(ring['radius'])} pixels")

    return [{
        "center": (center_x, center_y),
        "outer_radius": outer_radius,
        "rings": scoring_rings
    }]
