import cv2
import numpy as np
import os

INPUT_DIR = 'debug_output'
OUTPUT_DIR = 'refined_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def refine_center(image, circle):
    x, y, r = circle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, thickness=-1)

    # Canny edges inside circle
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Find contours in the edge mask
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (x, y), image  # No better center found

    # Combine all contour points into one set
    all_points = np.vstack(contours).squeeze()
    M = cv2.moments(all_points)
    if M["m00"] == 0:
        return (x, y), image  # Avoid division by zero

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    output = image.copy()
    cv2.circle(output, (x, y), 3, (0, 0, 255), -1)     # Original center (Red)
    cv2.circle(output, (cx, cy), 3, (0, 255, 255), -1) # Refined center (Yellow)
    cv2.line(output, (x, y), (cx, cy), (255, 0, 0), 1) # Show correction line

    # Optionally recenter the circle visually
    cv2.circle(output, (cx, cy), r, (0, 255, 0), 1)
    cv2.putText(output, "Refined", (cx - 30, cy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return (cx, cy), output

def main():
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".jpg"):
            continue
        path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(path)
        if image is None:
            print(f"Could not read: {path}")
            continue

        # Try to extract circle from filename
        parts = filename.split('_')
        try:
            r = 140  # If not passed, use a rough radius (adjust per your calibration)
            x = int(parts[-4])
            y = int(parts[-3])
        except:
            print(f"Skipping file (couldn’t extract circle center): {filename}")
            continue

        _, refined_img = refine_center(image, (x, y, r))
        out_path = os.path.join(OUTPUT_DIR, f"refined_{filename}")
        cv2.imwrite(out_path, refined_img)
        print(f"✅ Refined: {filename}")

if __name__ == "__main__":
    main()
