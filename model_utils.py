import numpy as np
import cv2
from tensorflow.keras.models import load_model
from config import MODEL_PATH, IMAGE_SIZE

model = load_model(MODEL_PATH)

def preprocess_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, IMAGE_SIZE)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def predict_image(image_path):

    img = preprocess_image(image_path)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        label = "Malignant"
    else:
        label = "Benign"

    confidence = float(prediction)

    return label, confidence