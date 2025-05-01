import argparse
import cv2
from detector.target_detector import detect_3spot_faces
from detector.arrow_detector import detect_arrows

def main(image_path, count_center_x_as_11=False, debug=True):
    # Load the image
    img = cv2.imread(image_path)

    # Step 1: Detect the target face(s)
    faces = detect_3spot_faces(img, debug=debug, count_center_x_as_11=count_center_x_as_11)

    if not faces:
        print("No target faces detected.")
        return

    # Step 2: For each detected target, detect arrows and score them
    for face in faces:
        print(f"Detected a target at {face['center']} with outer radius {face['outer_radius']} pixels.")
        
        for ring in face["rings"]:
            print(f" - Scoring Ring: Score {ring['score']} at radius {int(ring['radius'])} pixels")

        # Step 3: Detect arrows and their scores
        arrows = detect_arrows(img, face["center"], face["rings"], debug=debug)
        print("Detected arrows and their scores:")
        for a in arrows:
            print(f" - Arrow at {a['position']} scored {a['score']}")

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Archery Scoring Detection")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--x11", action="store_true", help="Count center ring as 11")
    parser.add_argument("--debug", action="store_true", help="Show debug images")
    
    args = parser.parse_args()

    # Run the main function with the given arguments
    main(args.image, count_center_x_as_11=args.x11, debug=args.debug)
