from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from datetime import datetime
from orientation import detect_angle


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def detect_orientation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scores = {}

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = gray
        elif angle == 90:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(gray, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

        top = rotated[:rotated.shape[0]//2, :]
        bottom = rotated[rotated.shape[0]//2:, :]

        edge_top = np.sum(cv2.Canny(top, 50, 150))
        edge_bottom = np.sum(cv2.Canny(bottom, 50, 150))

        scores[angle] = edge_bottom - edge_top

    best_angle = max(scores, key=scores.get)
    return best_angle


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["image"]

        filename = datetime.now().strftime("%H%M%S") + "_" + file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        image = cv2.imread(path)
        #detected = detect_orientation(image)
        detected = detect_angle(image)


        rotation_fix = {
            0: image,
            90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),          # correct
            180: cv2.rotate(image, cv2.ROTATE_180),
            270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)   # correct
            }

        corrected = rotation_fix[detected]

        corrected_name = "corrected_" + filename
        corrected_path = os.path.join(UPLOAD_FOLDER, corrected_name)
        cv2.imwrite(corrected_path, corrected)

        return render_template(
            "index.html",
            original=path,
            corrected=corrected_path,
            angle=detected
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
