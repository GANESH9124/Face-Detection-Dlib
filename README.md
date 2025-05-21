# Face Detection using Dlib and OpenCV

This project demonstrates how to perform face detection and facial landmark extraction using the Dlib and OpenCV libraries in Python. The primary goal is to identify faces in an image and then pinpoint 68 key facial landmarks on each detected face.


## 🌟 Key Features

*   **Face Detection:** Utilizes OpenCV's pre-trained Deep Neural Network (DNN) based face detector (Caffe model).
*   **Facial Landmark Extraction:** Employs Dlib's pre-trained shape predictor model to find 68 facial landmarks.
*   **Visualization:** Uses OpenCV and Matplotlib to display the original image with detected faces and landmarks.
*   **Modular Code:** Functions for detection and visualization are separated for clarity.

## 🛠️ Technologies Used

*   **Python 3.x**
*   **Dlib:** For facial landmark detection.
*   **OpenCV (cv2):** For image processing, DNN face detection, and drawing.
*   **NumPy:** For numerical operations, especially array manipulations.
*   **Matplotlib:** For displaying images within the Jupyter Notebook.
*   **Jupyter Notebook:** For interactive development and presentation.

## 🧠 Models Used

This project relies on two pre-trained models:

1.  **`res10_300x300_ssd_iter_140000.caffemodel` (and `deploy.prototxt`)**
    *   **Type:** Pre-trained face detection model (OpenCV DNN using Caffe).
    *   **Purpose:** Detects face bounding boxes in images.
    *   **Source:** Often distributed with OpenCV or can be downloaded from official OpenCV GitHub repositories. You will need both the `.caffemodel` (weights) and `.prototxt` (architecture) files.

2.  **`shape_predictor_68_face_landmarks.dat`**
    *   **Type:** Pre-trained facial landmark detector model for Dlib.
    *   **Purpose:** Detects 68 key facial landmarks within a detected face.
    *   **Source:** Downloadable from the [Dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). (You'll need to decompress the `.bz2` file).

## ⚙️ Setup & Installation

1.  **Prerequisites:**
    *   Python 3.x and pip installed.
    *   **CMake:** Dlib requires CMake to be installed for `pip install dlib` to work.
        *   **Windows:** Download from [cmake.org](https://cmake.org/download/) and add to PATH.
        *   **Linux (Ubuntu/Debian):** `sudo apt-get install cmake`

2.  **Install Python Libraries:**
    ```bash
    pip install opencv-python
    pip install numpy
    pip install matplotlib
    pip install dlib
    ```
    *   **Dlib Installation Notes (from notebook):**
        *   **Windows (CMD):**
            ```bash
            pip install cmake
            pip install dlib
            ```
        *   **Linux (Ubuntu/Debian Terminal):**
            ```bash
            sudo apt-get install cmake
            sudo apt-get install build-essential
            sudo apt-get install python3-dev
            pip install dlib
            ```

3.  **Download Model Files:**
    *   Download `shape_predictor_68_face_landmarks.dat` (e.g., from [Dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and decompress).
    *   Obtain `res10_300x300_ssd_iter_140000.caffemodel` and `deploy.prototxt`. These are standard OpenCV DNN face detection models. You can often find them in the OpenCV GitHub repository or related computer vision resources.

4.  **Project Structure:**
    Organize your project files as follows (the notebook assumes models are in `../models/` relative to the notebook's location):
    ```
    face-detection-dlib-opencv/
    ├── models/
    │   ├── shape_predictor_68_face_landmarks.dat
    │   ├── res10_300x300_ssd_iter_140000.caffemodel
    │   ├── deploy.prototxt
    ├── face_detection_dlib.ipynb
    ├── test_image.jpg  (or your test image)
    └── readme.md
    ```
    If your notebook is in the root `face-detection-dlib-opencv/` directory, then the `../models/` path in the notebook should be changed to `./models/` or simply `models/`. Adjust the paths in the notebook (`LANDMARK_MODEL_PATH`, `FACE_DETECTOR_MODEL_PATH`, `MODEL_DEFINITION_PATH`) to match your structure.

## 🚀 How to Run

1.  **Ensure all dependencies and models are correctly set up** as described above.
2.  **Place your test image** (e.g., `test_image.jpg`) in the project directory or update the path in the notebook.
3.  **Open and run the Jupyter Notebook:**
    ```bash
    jupyter notebook face_detection_dlib.ipynb
    ```
4.  Execute the cells in the notebook. The last cell will read the test image, perform detection and landmark extraction, and then visualize the result.

## 📜 Code Explanation

*   **Model Loading:**
    *   `cv2.dnn.readNetFromCaffe(MODEL_DEFINITION_PATH, FACE_DETECTOR_MODEL_PATH)`: Loads the OpenCV DNN face detector.
    *   `dlib.shape_predictor(LANDMARK_MODEL_PATH)`: Loads the Dlib facial landmark predictor.
*   **`DetectFaces(image)` Function:**
    1.  Takes an image (NumPy array) as input.
    2.  Creates a `blob` from the image suitable for the OpenCV DNN model (`cv2.dnn.blobFromImage`). This involves resizing and mean subtraction.
    3.  Passes the blob through the network (`net.setInput(blob)`, `net.forward()`) to get detections.
    4.  Iterates through detections, filtering by confidence (e.g., > 0.5).
    5.  For each confident detection:
        *   Calculates the bounding box coordinates.
        *   Creates a `dlib.rectangle` object for the face region.
        *   Converts the image to grayscale (Dlib shape predictor prefers grayscale).
        *   Uses `shape_predictor(gray_img, dlib_rect)` to get the 68 facial landmarks.
        *   Adjusts the bounding box around the landmarks with padding for better cropping.
    6.  Returns a list of dictionaries, each containing the `box`, `landmarks`, and `confidence` for a detected face.
*   **`VisualiseFaces(image, detection_result)` Function:**
    1.  Takes the original image and the `detection_result` from `DetectFaces`.
    2.  If faces are detected, it iterates through them.
    3.  Draws a green rectangle around the adjusted face box using `cv2.rectangle`.
    4.  Displays the image with detections using `matplotlib.pyplot`.

## 💡 Further Applications

This face detection and landmark extraction pipeline can be a preprocessing step for various applications, including:

*   Emotion Recognition
*   Face Verification/Recognition
*   Eye Tracking / Gaze Estimation
*   Head Pose Estimation
*   Virtual Makeup/Filter Applications

---

Feel free to modify this README to better suit any specific nuances of your project or if you add more features!
