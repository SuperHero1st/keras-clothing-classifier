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
TEST_SIZE = 0.15

def main():
    # Check command-line arguments

    if len(sys.argv) not in [2, 5]:
        sys.exit("Usage: python fashion_model_2nd_training.py mnist_dataset/train mnist_dataset/test new_model.v2 new_model.v3")

    # Load training and testing data
    x_train, y_train = load_data(sys.argv[1])
    x_test, y_test = load_data(sys.argv[2])

    # Convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train, dtype=int)
    x_test = np.array(x_test)
    y_test = np.array(y_test, dtype=int)

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    if os.path.exists(sys.argv[3]):
        print(f"Loading model from {sys.argv[3]}")
        model = tf.keras.models.load_model(sys.argv[3])
    else:
        model = get_model_improved()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 5:
        filename = sys.argv[4]
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
                    # Resize
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                    # Normalize
                    img = img.astype(np.float32) / 255.0
                    # Optionally print shape for debugging
                    # print(f"Image resized and normalized. Shape: {img.shape}")
                    images.append(img)
                    directories.append(int(directory))
    
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
        64, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

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


def get_model_improved():
    """
    Returns a compiled convolutional neural network model, aiming for better performance.
    """
    model = tf.keras.models.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(
            32, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Optional: tf.keras.layers.Dropout(0.25), # Dropout after pooling

        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Optional: tf.keras.layers.Dropout(0.25),

        # Block 3 (Optional, add if images are larger or more complex)
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Optional: tf.keras.layers.Dropout(0.25),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Dense Block
        tf.keras.layers.Dense(512), # Or 512 if more capacity is needed
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3), # Increased dropout

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adam is good, can try SGD too
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    main()
