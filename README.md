# Dysarthria Classification from .wav Files

This repository contains a Jupyter Notebook for classifying dysarthria and non-dysarthria speech from `.wav` files. The model is trained and saved as `model.h5`, and a FastAPI endpoint is provided for easy inference.


## Project Overview
Dysarthria is a motor speech disorder resulting from neurological injury that affects the muscles used in speech production. This project aims to classify audio files into dysarthria and non-dysarthria categories using machine learning.

## Installation
Clone the repository and navigate to the project directory:
git clone https://github.com/yourusername/dysarthria-classification.git
cd dysarthria-classification



## Usage
### Model Training
Open the `dysarthria.ipynb` Jupyter Notebook and run through the cells to preprocess the data, train the model, and save it as `model.h5`.

### FastAPI Deployment
1. Ensure `model.h5` is in the project directory.
2. Run the FastAPI server:
```bash
uvicorn main:app --reload
```
3. You can now make POST requests to the `/predict` endpoint with `.wav` files to get predictions.

## Model Training
The Jupyter Notebook contains the following steps:

### Data Visualization and EDA

*   Visualization Functions: Features and Plots
    *   Waveplot
    *   Spectrogram
    *   Zero Crossing Rate
    *   Spectral Centroids
    *   Spectral Rolloff
    *   MFCCs
    *   Mel Spectrogram

### Fetching Audio Samples

*   Male and Dysarthric
*   Female and Dysarthric
*   Male and Non-Dysarthric
*   Female and Non-Dysarthric

### Waveplots

Visualize the waveforms of audio samples.

### Spectrograms

Visualize the spectrograms of audio samples.

### Zero Crossing Rate

Calculate and visualize the zero crossing rate of audio samples.

### Spectral Centroid

Calculate and visualize the spectral centroid of audio samples.

### Spectral Rolloff

Calculate and visualize the spectral rolloff of audio samples.

### MFCCs

Extract and visualize the Mel-frequency cepstral coefficients (MFCCs) of audio samples.

### Mel Spectrogram

Extract and visualize the Mel spectrogram of audio samples.

### Feature Extraction

Extract features from the audio samples for model training.

### Modelling

Train a neural network using Keras.

### Model Summary

Display the model summary and learning curves.

### Model Evaluation

*   ROC Curve and AUC Score
*   Confusion Matrix
    *   Only 1 misclassification from each class is observed.
    *   Upon running the model several times, the performance remains consistent.
*   Classification Report

## FastAPI Deployment
The `app.py` file contains the FastAPI code to serve the model. The endpoint accepts `.wav` files, processes them, and returns the classification result.

## Results
The trained model achieves an accuracy of 93% on the test set. Detailed results and performance metrics are available in the Jupyter Notebook.
