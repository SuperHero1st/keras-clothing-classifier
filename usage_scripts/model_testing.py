import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# --- Configuration ---
MODEL_PATH = 'A:/AI model/new_model.v2' # SavedModel directory
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 60          # Original model's input dimensions
EXPECTS_GRAYSCALE = False                    # True if original model used grayscale
CLASS_NAMES = ['pants', 'long sleeve', 'dress', 'bags', 'footwear']
NEW_IMAGE_PATH = "C:/Users/ahmed/Desktop/bag.jpg" # Path to your new image

# 1. Load the saved model
model = keras.models.load_model(MODEL_PATH)

# 2. Load and preprocess the new image using OpenCV
img = cv2.imread(NEW_IMAGE_PATH)

# Resize
img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
print(f"Image resized. Shape after resize: {img.shape}")

# Normalize
img_array = np.array(img) / 255.0 # Convert to float and normalize

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0) # (H, W, C) -> (1, H, W, C)
print(f"Final preprocessed image array shape: {img_array.shape}")

# 3. Make prediction
predictions = model.predict(img_array)
print(predictions)
predicted_class_index = np.argmax(predictions[0])
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = np.max(predictions[0])

# 4. Display result
print(f"Image: {NEW_IMAGE_PATH}")
print(f"Predicted class: {predicted_class_name} (Index: {predicted_class_index})")
print(f"Confidence: {confidence:.4f}")