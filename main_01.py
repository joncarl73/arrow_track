from detector.target_detector import detect_3spot_faces
import cv2

def main(image_path):
    img = cv2.imread(image_path)
    faces = detect_3spot_faces(img, debug=True)
    print(f"Detected {len(faces)} target faces.")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
