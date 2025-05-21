# app.py
from flask import Flask, request, jsonify, render_template 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

MODEL_PATH = 'A:/AI model/fashion_classifier'
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 60
CLASS_NAMES = ['pants', 'long sleeve', 'dress', 'bags', 'footwear']

app = Flask(__name__)

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# New route to serve the HTML page
@app.route("/")
def index():
    return render_template("index.html") # Assumes index.html is in a 'templates' folder

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image file stream into a numpy array using OpenCV
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Reads as BGR
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = image / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions_raw = model.predict(img_array) # This is usually an array of arrays like [[0.1, 0.7, 0.05, ...]]
        
        # Get the probabilities for the single image
        probabilities = predictions_raw[0] 

        predicted_class_index = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        if predicted_class_index >= len(CLASS_NAMES):
            return jsonify({'error': f'Predicted class index {predicted_class_index} is out of bounds for CLASS_NAMES.'}), 500

        predicted_label = CLASS_NAMES[predicted_class_index]

        # Create the all_confidences dictionary
        all_confidences_dict = {
            CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
        }
        # Ensure the number of probabilities matches the number of class names
        if len(probabilities) != len(CLASS_NAMES):
            # Log this error on the server, it's a mismatch between model output and CLASS_NAMES
            app.logger.error(f"Mismatch between number of probabilities ({len(probabilities)}) and CLASS_NAMES ({len(CLASS_NAMES)})")
            all_confidences_dict = {} # Send empty if there's a mismatch


        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence, 4),
            'class_index': predicted_class_index, 
            'all_confidences': all_confidences_dict
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
