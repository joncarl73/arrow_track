import argparse
import cv2
from detector.target_detector import detect_3spot_faces
from detector.arrow_detector import detect_arrows
from detector.hole_detector import detect_holes  # New import

def main(image_path, count_center_x_as_11=False, debug=True, detection_mode="arrow"):
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Cannot load image at {image_path}")
        return

    # Step 1: Detect the target face(s)
    faces = detect_3spot_faces(img, debug=debug, count_center_x_as_11=count_center_x_as_11)

    if not faces:
        print("No target faces detected.")
        return

    # Step 2: For each detected target, apply the selected detection mode
    for face in faces:
        print(f"Detected a target at {face['center']} with outer radius {face['outer_radius']} pixels.")
        
        for ring in face["rings"]:
            print(f" - Scoring Ring: Score {ring['score']} at radius {int(ring['radius'])} pixels")

        # Step 3: Detect arrows or holes and print scores
        if detection_mode == "hole":
            print("Using hole detection...")
            results = detect_holes(img, face["center"], face["rings"], debug=debug)
        else:
            print("Using arrow detection...")
            results = detect_arrows(img, face["center"], face["rings"], debug=debug)

    print("Detected shots and their scores:")
    for idx, res in enumerate(results):
        label = chr(65 + idx)  # 65 is ASCII for 'A'
        print(f" {label}: Shot at {res['position']} scored {res['score']}")

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Archery Scoring Detection")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--x11", action="store_true", help="Count center ring as 11")
    parser.add_argument("--debug", action="store_true", help="Show debug images")
    parser.add_argument("--mode", choices=["arrow", "hole"], default="arrow",
                        help="Choose detection mode: 'arrow' (default) or 'hole'")

    args = parser.parse_args()

    # Run the main function with CLI args
    main(args.image, count_center_x_as_11=args.x11, debug=args.debug, detection_mode=args.mode)
