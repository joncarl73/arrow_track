import cv2
import numpy as np

def detect_arrows(img, target_center, rings, debug=False):
    arrows = []
    
    # Get the outermost ring radius (the largest ring)
    outer_radius = rings[-1]["radius"]

    # Convert the image to grayscale and apply adaptive thresholding to isolate arrows
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Gaussian blur to reduce noise
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the thresholded image (potential arrow shapes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_arrows = []
    
    for cnt in contours:
        # Ignore small contours (noise)
        if cv2.contourArea(cnt) < 100:
            continue
        
        # Get the arrow tip using the bounding box method
        arrow_tip = get_arrow_tip_from_bounding_box(cnt)

        if arrow_tip:
            # Check if the arrow tip is within the target's circular boundary (outermost ring)
            distance_from_center = np.linalg.norm(np.array(arrow_tip) - np.array(target_center))
            if distance_from_center <= outer_radius:
                # Calculate the entry point of the arrow based on the arrow tip position
                score = calculate_arrow_score(arrow_tip, target_center, rings)
                detected_arrows.append({"position": arrow_tip, "score": score})

    # If there are multiple detected arrows, filter out based on closest to the target center
    if len(detected_arrows) > 1:
        detected_arrows = [min(detected_arrows, key=lambda x: np.linalg.norm(np.array(x["position"]) - np.array(target_center)))]

    # Debugging visualization
    if debug:
        for arrow in detected_arrows:
            # Highlight the arrow tip with a red circle
            cv2.circle(img, arrow["position"], 10, (0, 0, 255), 3)  # Red for detected arrow tip
            
            # Display the arrow tip with the label "A"
            cv2.putText(img, "A", arrow["position"], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw the bounding box around the detected contour
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for bounding box

        cv2.imshow("Detected Arrows", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detected_arrows

def get_arrow_tip_from_bounding_box(cnt):
    """ Get the tip of the arrow from the contour's bounding box """
    # Get the bounding box for the contour (x, y, w, h)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # We assume that the tip of the arrow is at the point farthest along the shaft
    # For now, let's take the center of the bounding box as a proxy for the tip
    arrow_tip = (x + w // 2, y + h // 2)
    
    return arrow_tip

def get_arrow_tip_by_furthest_point(cnt, target_center):
    """ Return the point in the contour that is furthest from the target center """
    max_dist = 0
    tip_point = None
    for point in cnt:
        pt = point[0]  # unwrap [[x, y]] to [x, y]
        dist = np.linalg.norm(np.array(pt) - np.array(target_center))
        if dist > max_dist:
            max_dist = dist
            tip_point = tuple(pt)
    return tip_point


def calculate_arrow_score(arrow_tip, target_center, rings):
    """ Calculate score based on where the arrow tip lies relative to the rings """
    distance_from_center = np.linalg.norm(np.array(arrow_tip) - np.array(target_center))
    
    for ring in rings:
        buffer_zone = 2  # Allow a buffer around each ring's boundary
        if distance_from_center <= ring["radius"] + buffer_zone:
            return ring["score"]
    
    return 0  # Default score if no valid ring is hit
