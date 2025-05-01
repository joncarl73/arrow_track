import cv2
import os
import numpy as np
import sys

def refine_center_by_edges(image, x, y, r, debug_dir=None, debug_prefix=""):
    """Refines the (x, y) center of a circle using edge density inside the radius."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    med_val = np.median(gray)
    lower = int(max(0, 0.66 * med_val))
    upper = int(min(255, 1.33 * med_val))
    edges = cv2.Canny(gray, lower, upper)

    # Expand radius slightly
    r_expand = int(r * 1.05)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r_expand, 255, -1)

    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Debug image output
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_gray.jpg"), gray)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_edges.jpg"), edges)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_masked_edges.jpg"), masked_edges)

    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"[WARN] No contours found for {debug_prefix} — using fallback center.")
        h, w = image.shape[:2]
        return w // 2, h // 2

    all_points = np.vstack(contours).squeeze()
    if all_points.ndim != 2 or all_points.shape[0] < 10:
        print(f"[WARN] Insufficient contour points for {debug_prefix} — using fallback center.")
        h, w = image.shape[:2]
        return w // 2, h // 2

    M = cv2.moments(all_points)
    if M["m00"] == 0:
        print(f"[WARN] Zero division in moment for {debug_prefix} — using fallback center.")
        h, w = image.shape[:2]
        return w // 2, h // 2

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if debug_dir:
        debug_image = image.copy()
        cv2.circle(debug_image, (cx, cy), 5, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_center_debug.jpg"), debug_image)

    return cx, cy


def main(image_path, x, y, r, debug_dir="debug"):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return

    cx, cy = refine_center_by_edges(image, x, y, r, debug_dir=debug_dir, debug_prefix=os.path.basename(image_path).split('.')[0])
    print(f"Refined center for {image_path}: ({cx}, {cy})")

    # Optionally show result
    image_copy = image.copy()
    cv2.circle(image_copy, (cx, cy), 5, (0, 255, 0), -1)
    cv2.imshow("Refined Center", cv2.resize(image_copy, (300, 300)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python refine_center_debug.py <image_path> <x> <y> <r>")
        sys.exit(1)

    image_path = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    r = int(sys.argv[4])
    main(image_path, x, y, r)
