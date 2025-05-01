import cv2
import numpy as np

def hex_to_hsv(hex_color):
    bgr = np.uint8([[tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0]

def create_color_mask(hsv_img, hex_color, tolerance=(10, 100, 100)):
    target_hsv = hex_to_hsv(hex_color)
    h, s, v = [int(c) for c in target_hsv]

    lower = np.array([
        max(0, h - tolerance[0]),
        max(0, s - tolerance[1]),
        max(0, v - tolerance[2])
    ])
    upper = np.array([
        min(179, h + tolerance[0]),
        min(255, s + tolerance[1]),
        min(255, v + tolerance[2])
    ])
    mask = cv2.inRange(hsv_img, lower, upper)
    return mask


