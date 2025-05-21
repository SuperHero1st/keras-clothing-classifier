# Keras Clothing Classifier Web App

A Flask web application that allows users to upload an image of a clothing item. A pre-trained Keras AI model then classifies the item into categories like 'pants', 'long sleeve', 'dress', 'bags', or 'footwear', and displays the prediction with confidence scores.

<!-- Optional: Add a screenshot or GIF of your application in action -->
<!-- ![App Screenshot](https://github.com/SuperHero1st/keras-clothing-classifier/blob/main/app_view.jpg) -->

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Model Information](#model-information)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Contributing](#contributing)

## Features
- **Image Upload:** User-friendly interface to upload clothing images.
- **AI-Powered Classification:** Utilizes a Keras/TensorFlow model to predict the clothing category.
- **Confidence Scores:** Displays the overall confidence of the prediction.
- **Per-Category Probabilities:** Shows the model's confidence for each possible clothing type.
- **Web Interface:** Simple and intuitive web UI built with Flask and HTML.

## Technologies Used
- **Backend:** Python, Flask
- **Machine Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV (`cv2`), NumPy
- **Frontend:** HTML, (potentially CSS, JavaScript if you have a more complex `index.html`)
- **Version Control:** Git, Git LFS (for model file management)

## Model Information
- **Model Type:** Deep Learning model (likely a Convolutional Neural Network - CNN) built with Keras.
- **Dataset:** (Specify the dataset used for training, e.g., Fashion-MNIST, DeepFashion, or "a custom dataset of clothing images").
- **Input Image Size:** Expected input images are resized to `60x80` pixels (Width x Height).
- **Class Categories:** The model classifies items into the following categories:
    - `pants`
    - `long sleeve`
    - `dress`
    - `bags`
    - `footwear`
- **Model File:** The pre-trained model is stored in the repository (e.g., `models/fashion_classifier/`) and managed using Git LFS.

## Setup and Installation

**Prerequisites:**
- Python 3.7+
- pip (Python package installer)
- Git
- **Git LFS:** You *must* have Git LFS installed to properly clone and pull the large model files. Download and install it from [git-lfs.github.com](https://git-lfs.github.com/). After installing, run `git lfs install` once in your terminal.

**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[SuperHero1st]/keras-clothing-classifier.git
    cd keras-clothing-classifier
    ```
    *(Git LFS should automatically download the model files during the clone process if you have it installed correctly.)*

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify Model Files (if LFS was used):**
    Ensure the model files are present in the `models/fashion_classifier/` directory (or the path specified in `app.py` if you adjust it to be relative). If they are small pointer files, you might need to run `git lfs pull` after installing Git LFS.

## Running the Application

1.  **Navigate to the project directory** (if you're not already there).
2.  **Ensure your virtual environment is activated.**
3.  **Run the Flask application:**
    ```bash
    python app.py
    ```
4.  **Open your web browser** and go to: `http://127.0.0.1:5000/` (or `http://localhost:5000/`)

You should see the web interface where you can upload an image for classification.

## How It Works
1.  The user navigates to the web application in their browser.
2.  The Flask backend serves the `index.html` page.
3.  The user selects an image file and submits it through the form.
4.  The `/predict` endpoint in `app.py` receives the image.
5.  The image is preprocessed:
    - Read into a NumPy array using OpenCV.
    - Resized to the model's expected input dimensions (`60x80`).
    - Normalized (pixel values divided by 255.0).
    - Expanded to match the batch input shape for the model.
6.  The pre-trained Keras model (`model.predict()`) processes the image array and outputs prediction probabilities for each class.
7.  The application determines the predicted class with the highest probability and also prepares a dictionary of all class probabilities.
8.  The results (predicted label, main confidence, and all class confidences) are returned as a JSON response to the frontend, which then displays them to the user.

<!-- Optional Sections -->
<!--
## Future Improvements
- Add support for more clothing categories.
- Improve model accuracy.
- Implement batch image processing.
- Deploy to a cloud platform (e.g., Heroku, AWS, Google Cloud).
-->

## Contributing
Contributions are welcome! If you have suggestions for improvements or find any bugs, please feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
