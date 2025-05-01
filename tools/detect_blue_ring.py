import cv2
import numpy as np
import sys
import os

def detect_blue_ring(image_path, debug_dir="debug"):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None

    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define blue range in HSV (tweak as needed)
    lower_blue = np.array([90, 80, 50])   # hue, sat, val
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological clean up
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[WARN] No blue contours found.")
        return None

    # Find the largest contour (assumed to be the blue ring)
    largest = max(contours, key=cv2.contourArea)

    # Fit a minimum enclosing circle to the blue ring
    (x, y), radius = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius)

    # Draw circle and center on a copy
    debug_img = image.copy()
    cv2.circle(debug_img, center, radius, (255, 0, 0), 2)   # Blue ring
    cv2.circle(debug_img, center, 3, (0, 0, 255), -1)        # Center dot

    os.makedirs(debug_dir, exist_ok=True)
    base = os.path.basename(image_path).split('.')[0]
    out_path = os.path.join(debug_dir, f"{base}_blue_ring_detected.jpg")
    cv2.imwrite(out_path, debug_img)
    print(f"[INFO] Saved debug image to {out_path}")

    return center, radius

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_blue_ring.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_blue_ring(image_path)
