import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 5
IMG_WIDTH = 60
IMG_HEIGHT = 80
NUM_CATEGORIES = 5
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python fashion_model.py 'A:/AI model/2nd_dataset' [model.h5]")

    # Load all data from the directory
    images, labels = load_data(sys.argv[1])

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels, dtype=int)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, random_state=42
    )

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CATEGORIES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CATEGORIES)

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    components = data_dir.split(os.sep if os.sep in data_dir else '/')
    # Join the components using os.path.join
    normalized_path = os.path.join(*components)
    normalized_path = data_dir

    images = []
    directories = []
    
    for directory in os.listdir(normalized_path):
        directory_path = os.path.join(normalized_path, directory)
        print(directory)
        
        if os.path.isdir(directory_path):
            for image in os.listdir(directory_path):
                image_path = os.path.join(directory_path, image)
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(img)
                    directories.append(directory)
    
    print(f"Loaded {len(images)} images and {len(directories)} labels from {data_dir}")  # <-- Add this line
    return (images, directories)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

    # Convolutional layer. Learn 128 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),

    # Flatten units
    tf.keras.layers.Flatten(),

    # Add a hidden layer with dropout
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.05),

    # Add an output layer with output units for all 10 digits
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
