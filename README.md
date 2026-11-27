# Pneumonia Detection Using Deep Learning and Convolutional Neural Networks

## Project Overview

This project focuses on the development of a deep learning-based system for the early detection of pneumonia from chest X-ray images.

It uses a customised Convolutional Neural Network (CNN), called **PneumoCNN**, and compares its performance with four established transfer-learning architectures:

- VGG16
- ResNet50
- DenseNet121
- InceptionV3

The models classify chest X-ray images into two categories:

- **NORMAL:** No pneumonia detected by the model.
- **PNEUMONIA:** Features associated with pneumonia detected by the model.

The project also includes a Streamlit web application that allows users to upload chest X-ray images and obtain model predictions.

**Note:** This is an academic research project and is not intended to replace professional medical diagnosis.

---

## Objectives

1. Develop a customised CNN architecture for pneumonia detection.
2. Implement transfer-learning models using VGG16, ResNet50, DenseNet121 and InceptionV3.
3. Compare the performance of the five models.
4. Evaluate the models using accuracy, precision, sensitivity, specificity, F1-score and ROC-AUC.
5. Develop a user-friendly web application using Streamlit.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network development |
| NumPy | Numerical computations |
| Pandas | Data manipulation |
| OpenCV | Image processing |
| Matplotlib | Data visualisation |
| Scikit-learn | Model evaluation |
| Streamlit | Web application |
| Google Colab | Model training and experimentation |
| Jupyter Notebook | Development and experimentation |

---

## Models Implemented

### 1. Custom CNN (PneumoCNN)

The customised CNN consists of four convolutional blocks with 32, 64, 128 and 256 filters respectively.

Each block uses:
- Convolutional layer
- Batch Normalisation
- ReLU activation
- Max Pooling

The classification head includes Global Average Pooling, a 128-unit dense layer, Dropout and a sigmoid output layer.

### 2. Transfer Learning Models

The following pretrained architectures were implemented using ImageNet weights:

- VGG16
- ResNet50
- DenseNet121
- InceptionV3

The pretrained feature extraction layers were frozen during the experiments, while additional classification layers were trained for pneumonia detection.

---

## Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

**Dataset source:**

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains two image categories:

- NORMAL
- PNEUMONIA

### Dataset Structure

Organise the dataset as follows:

```text
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The project also uses cleaned CSV split files for the training, validation and independent test datasets.

---

## Project Structure

```text
pneumonia-detection-deep-learning/
│
├── app/
│   ├── main.py
│   └── util.py
│
├── model/
│   └── pneumonia_classifier.h5
│
├── notebooks/
│   └── pneumonia_detection.ipynb
│
├── data/
│   └── chest_xray/
│
├── training_histories/
│
├── evaluation_results/
│
├── requirements.txt
│
├── labels.txt
│
└── README.md
```

*Your actual folder structure may differ. Adjust the paths to match the files in your repository.*

---

## Installation and Setup

### Step 1: Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Navigate into the project folder:

```bash
cd pneumonia-detection-deep-learning
```

### Step 2: Install Python

Install Python 3.10.

For the TensorFlow 2.12 environment used in this project, Python 3.10 is recommended.

Check your Python installation:

```bash
python --version
```

### Step 3: Create a Virtual Environment

On Windows:

```bash
python -m venv med_env
```

Activate the environment:

```bash
med_env\Scripts\activate
```

### Step 4: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 5: Install Dependencies

Create a `requirements.txt` file containing:

```text
tensorflow==2.12.0
numpy==1.23.5
pandas
matplotlib
scikit-learn
opencv-python
Pillow
streamlit
seaborn
```

Install the packages:

```bash
pip install -r requirements.txt
```

**Important:** TensorFlow 2.12 requires a compatible Python environment. Avoid using Python 3.14 or 3.15 for this environment.

---

## Running the Application

After installing the dependencies, ensure that the trained model is available at:

```text
model/pneumonia_classifier.h5
```

Also ensure that `labels.txt` and the application files are in their expected locations.

Run the Streamlit application:

```bash
streamlit run app/main.py
```

If your `main.py` file is in the project root, use:

```bash
streamlit run main.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

### How to Use the Application

1. Open the Streamlit application.
2. Upload a chest X-ray image.
3. Allow the application to process the image.
4. View the predicted classification.
5. Review the model's prediction probabilities.

---

## Model Evaluation

The models are evaluated using the following performance metrics:

| Metric | Description |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of positive predictions that are correct |
| Sensitivity | Ability to identify pneumonia cases |
| Specificity | Ability to identify normal cases |
| F1-score | Harmonic mean of precision and sensitivity |
| ROC-AUC | Ability to distinguish between the two classes |

The models are compared using the same cleaned independent test dataset.

---

## Research Contribution

The main contribution of this research is the development and evaluation of a customised CNN architecture (PneumoCNN) and its comparison with four established transfer-learning architectures.

The research investigates how a purpose-built CNN performs against pretrained models when classifying chest X-ray images into normal and pneumonia categories.

---

## Limitations

- Model predictions depend on the quality and characteristics of the input images.
- False-positive and false-negative predictions may occur.
- Performance on the research dataset does not guarantee performance on images from other hospitals or populations.
- The system has not been clinically validated and should not be used as a standalone diagnostic tool.

---

## Future Improvements

- Improve the specificity of the models.
- Investigate techniques for reducing false-positive predictions.
- Conduct further experiments with data augmentation.
- Explore explainable AI techniques such as Grad-CAM.
- Evaluate the models using external datasets.
- Investigate deployment in real-world healthcare research environments.

---

## Author

**Bukari Amo**

MPhil Computer Science Research Project

### Project Title

*Early Detection of Pneumonia Using Convolutional Neural Networks and Deep Learning*

---

## Acknowledgements

Special appreciation to the research supervisor and the open-source deep learning community for the tools, frameworks and resources that supported this research.

## Disclaimer

This application is developed strictly for academic and research purposes. It is not a substitute for professional medical advice, diagnosis or treatment.
