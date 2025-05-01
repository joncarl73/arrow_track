import cv2
import numpy as np
import os

INPUT_DIR = 'data/targets'
OUTPUT_DIR = 'debug_output'
IMAGES = [f'target_face_{i}.jpg' for i in range(1, 4)]

PARAM2_VALUES = [40, 35, 30, 25, 20]
MIN_RADIUS = 110
MAX_RADIUS = 160

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fallback_contour_detection(img, img_gray):
    # Edge detection
    edged = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    found = False

    for cnt in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        if MIN_RADIUS < radius < MAX_RADIUS:
            cv2.circle(output, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), 3)
            cv2.putText(output, "Target (Contour)", (int(x - 50), int(y - radius - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            found = True
            break  # Only take the best one

    return output if found else None

def detect_and_save(img_path, param2_val):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Cannot read: {img_path}")
        return

    img = cv2.resize(img, (300, 300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    output = img.copy()

    # Try HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=param2_val,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS
    )

    method = "Hough"
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx, (x, y, r) in enumerate(circles[0, :]):
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
            cv2.putText(output, f"Target ({method})", (x - 50, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        fallback = fallback_contour_detection(img, gray)
        if fallback is not None:
            output = fallback
            method = "Contour"
        else:
            print(f"⚠️  No circles detected in {img_path} with param2={param2_val} using either method.")
            return

    out_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_p2_{param2_val}_{method}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), output)

def main():
    for img_name in IMAGES:
        img_path = os.path.join(INPUT_DIR, img_name)
        for p2 in PARAM2_VALUES:
            print(f"🔍 Testing {img_name} with param2={p2}")
            detect_and_save(img_path, p2)

    print("✅ Done. Check 'debug_output/' for results.")

if __name__ == "__main__":
    main()
