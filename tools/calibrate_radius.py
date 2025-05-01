import cv2
import numpy as np
import sys

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Calibration", param)
        if len(points) == 2:
            finalize(param)

def finalize(img):
    global points
    pt1, pt2 = points
    distance = np.linalg.norm(np.array(pt1) - np.array(pt2)) / 2
    print(f"\nEstimated Radius: {distance:.1f} pixels")
    print(f"Suggested minRadius: {int(distance * 0.85)}")
    print(f"Suggested maxRadius: {int(distance * 1.15)}")
    cv2.line(img, pt1, pt2, (0, 255, 255), 2)
    cv2.imshow("Calibration", img)

def main(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (300, 900), interpolation=cv2.INTER_AREA)
    clone = img_resized.copy()

    print("🔍 Click two points across the outer ring of a single target face.")
    print("💡 Tip: Click from one edge of the blue ring to the opposite edge.")
    cv2.imshow("Calibration", img_resized)
    cv2.setMouseCallback("Calibration", click_event, clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/calibrate_radius.py path/to/image.jpg")
    else:
        main(sys.argv[1])
