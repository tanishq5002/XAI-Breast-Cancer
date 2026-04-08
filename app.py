from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

from model_utils import predict_image
from gradcam_utils import generate_gradcam
from config import UPLOAD_FOLDER

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    # ✅ SECURE FILE NAME (IMPORTANT)
    filename = secure_filename(file.filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    print("Processing image:", filename)   # optional debug log

    label, confidence = predict_image(filepath)

    return render_template(
        "result.html",
        prediction=label,
        confidence=round(float(confidence), 4),
        image=filename
    )


# Metrics Dashboard Route
@app.route("/metrics")
def metrics():

    metrics_data = {
        "accuracy": 0.8055,
        "precision": 0.8213,
        "recall": 0.9161,
        "f1": 0.8661
    }

    return render_template(
        "metrics.html",
        metrics=metrics_data
    )


# GradCAM Explanation Route
@app.route("/explain", methods=["POST"])
def explain():

    image_name = request.form.get("image_name")
    prediction = request.form.get("prediction")
    confidence = request.form.get("confidence")

    if not image_name:
        return "Error: Image not found"

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_name)

    heatmap_path, overlay_path = generate_gradcam(image_path)

    heatmap_file = os.path.basename(heatmap_path)
    overlay_file = os.path.basename(overlay_path)

    return render_template(
        "explain.html",
        image=image_name,
        heatmap=heatmap_file,
        overlay=overlay_file,
        prediction=prediction,
        confidence=confidence
    )


# Optional: Handle unknown routes
@app.errorhandler(404)
def not_found(e):
    return "Page not found", 404


if __name__ == "__main__":
    app.run(debug=True)