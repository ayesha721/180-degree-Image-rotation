import cv2
import numpy as np

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    return gray / 255.0

def extract(gray):
    h, w = gray.shape
    top, bottom = gray[:h//2, :], gray[h//2:, :]

    edges = cv2.Canny((gray*255).astype(np.uint8), 100, 200) / 255.0
    proj = np.sum(gray, axis=1)

    score = 0
    if np.mean(top) > np.mean(bottom): score += 1
    if np.sum(edges[:h//2, :]) > np.sum(edges[h//2:, :]): score += 1
    if np.sum(proj[:h//2]) > np.sum(proj[h//2:]): score += 1
    return score

def detect_rotation(img):
    rotations = {
        0: img,
        90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(img, cv2.ROTATE_180),
        270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }

    best_angle = 0
    lowest = 999

    for angle, im in rotations.items():
        score = extract(preprocess(im))
        if score < lowest:
            lowest = score
            best_angle = angle

    corrected = rotations[best_angle]
    return best_angle, corrected
