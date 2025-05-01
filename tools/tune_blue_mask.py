import cv2
import numpy as np
import os

def nothing(x):
    pass

def tune_blue_mask(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return

    # Resize for viewing if needed
    img = cv2.resize(img, (600, 600))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("Blue Mask Tuner")
    cv2.createTrackbar("H Low", "Blue Mask Tuner", 90, 179, nothing)
    cv2.createTrackbar("H High", "Blue Mask Tuner", 130, 179, nothing)
    cv2.createTrackbar("S Low", "Blue Mask Tuner", 80, 255, nothing)
    cv2.createTrackbar("S High", "Blue Mask Tuner", 255, 255, nothing)
    cv2.createTrackbar("V Low", "Blue Mask Tuner", 50, 255, nothing)
    cv2.createTrackbar("V High", "Blue Mask Tuner", 255, 255, nothing)

    while True:
        h_low = cv2.getTrackbarPos("H Low", "Blue Mask Tuner")
        h_high = cv2.getTrackbarPos("H High", "Blue Mask Tuner")
        s_low = cv2.getTrackbarPos("S Low", "Blue Mask Tuner")
        s_high = cv2.getTrackbarPos("S High", "Blue Mask Tuner")
        v_low = cv2.getTrackbarPos("V Low", "Blue Mask Tuner")
        v_high = cv2.getTrackbarPos("V High", "Blue Mask Tuner")

        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])

        mask = cv2.inRange(hsv, lower, upper)

        # Optional: overlay mask on image for visualization
        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("Mask", mask)
        cv2.imshow("Masked Image", result)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # ESC or 'q' to quit
            break
        elif key == ord('s'):
            print(f"\n[INFO] Saved thresholds:")
            print(f"Lower HSV: {lower}")
            print(f"Upper HSV: {upper}")
            cv2.imwrite("blue_mask_debug.jpg", result)
            print("[INFO] Debug image saved as blue_mask_debug.jpg")

    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tune_blue_mask.py <image_path>")
        sys.exit(1)
    tune_blue_mask(sys.argv[1])
