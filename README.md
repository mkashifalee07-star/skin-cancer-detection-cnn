
# Skin Cancer Detection Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify
skin lesion images into nine categories.

The project was developed using Python, TensorFlow, Keras, and Kaggle.
A Streamlit application is included for image upload and prediction.

## Classes
1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

## Model
- Architecture: Custom CNN
- Input image size: 224 × 224
- Output classes: 9
- Optimizer: Adam
- Loss function: Categorical cross-entropy

## Evaluation
The current model achieved approximately **21.61% test accuracy**.

This result indicates poor classification performance. The model is
a learning project and requires substantial improvement before it
can be considered useful for research beyond experimentation.

## Project Files
- `app.py` — Streamlit interface
- `requirements.txt` — Required Python packages
- `skin_cancer_cnn.keras` — Saved trained model
- `best_skin_cancer_cnn.keras` — Best checkpoint from training
- `class_names.json` — Model class labels
- `test_metrics.json` — Test metrics
- `classification_report.csv` — Per-class evaluation results
- `confusion_matrix.png` — Confusion matrix visualization
- `accuracy_plot.png` — Training and validation accuracy
- `loss_plot.png` — Training and validation loss
- `sample_10_predictions.png` — Example predictions
- `sample_10_predictions.csv` — Example prediction results

## Run the App Locally
Install the required packages:

```bash
pip install -r requirements.txt
