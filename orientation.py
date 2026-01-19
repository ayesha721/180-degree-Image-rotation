import cv2
import numpy as np

def get_score(gray):
    h, w = gray.shape

    top = gray[:h//2, :]
    bottom = gray[h//2:, :]

    edges = cv2.Canny(gray, 50, 150)

    edge_top = np.sum(edges[:h//2, :])
    edge_bottom = np.sum(edges[h//2:, :])

    proj = np.sum(gray, axis=1)

    score = (edge_bottom - edge_top) - np.std(proj)
    return score


def detect_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256,256))
    rotations = {
    0: gray,
    90: cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
    180: cv2.rotate(gray, cv2.ROTATE_180),
    270: cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)  # added
    }

    scores = {angle: get_score(img) for angle, img in rotations.items()}
    best = max(scores, key=scores.get)

    return best
