import numpy as np
import cv2
import tensorflow as tf
import os

from tensorflow.keras.models import load_model
from config import MODEL_PATH, IMAGE_SIZE, HEATMAP_FOLDER, OVERLAY_FOLDER


model = load_model(MODEL_PATH)

# Force-build the model
dummy_input = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
model(dummy_input)


def preprocess_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, IMAGE_SIZE)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2"):

    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Create model that maps input → last conv layer output
    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    # Model from conv layer → predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:

        conv_outputs = conv_model(img_array)

        tape.watch(conv_outputs)

        predictions = classifier_model(conv_outputs)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap


def generate_gradcam(image_path):

    img_array = preprocess_image(image_path)

    heatmap = make_gradcam_heatmap(img_array, model)

    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)

    heatmap = cv2.resize(heatmap, IMAGE_SIZE)

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    filename = os.path.basename(image_path)

    heatmap_filename = "heatmap_" + filename
    overlay_filename = "overlay_" + filename

    heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
    overlay_path = os.path.join(OVERLAY_FOLDER, overlay_filename)

    cv2.imwrite(heatmap_path, heatmap_color)
    cv2.imwrite(overlay_path, overlay)

    return heatmap_path, overlay_path