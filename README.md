# Diabetes Risk Prediction using a Neural Network

**Author:** Geethaka09
**Date:** 2025-08-23

---

## 1. Project Overview

This project focuses on developing a robust machine learning model to predict an individual's risk of diabetes based on key health and lifestyle factors. The core of the project is a Neural Network built with TensorFlow and Keras, which has been carefully optimized for both accuracy and generalization.

The development process involved several key stages:
- Initial data loading and preprocessing.
- Building a baseline deep learning model.
- Implementing regularization techniques like `Dropout` and `EarlyStopping` to prevent overfitting.
- Optimizing the model's prediction threshold to achieve the best balance of precision and recall, which is critical for a medical diagnostic tool.

The final result is a trained and saved model capable of making reliable predictions on new, unseen data.

---

## 2. The Dataset

The project utilizes a clean, perfectly balanced dataset containing 346 patient records.

- **Target Variable:** `Diabetes` (0 for No, 1 for Yes)
- **Features (Predictors):**
  - `Age`: Age of the patient.
  - `Gender`: 0 for Female, 1 for Male.
  - `BMI`: Body Mass Index.
  - `PhysActivity`: Regular physical activity (0 for No, 1 for Yes).
  - `Smoker`: Whether the patient is a smoker (0 for No, 1 for Yes).
  - `HvyAlcoholConsump`: Heavy alcohol consumption (0 for No, 1 for Yes).
  - `Family`: Family history of diabetes (0 for No, 1 for Yes).

A key finding from the exploratory data analysis was that the dataset is balanced, which simplifies the training process as the model is not inherently biased towards one class.

---

## 3. Core Methodologies

The project implements a complete and effective machine learning workflow:

- **Data Preprocessing:** Numerical features (`Age`, `BMI`) were scaled to a common range using `MinMaxScaler` to ensure they contribute equally to model training.
- **Neural Network Architecture:** A sequential model was constructed using TensorFlow/Keras, featuring multiple dense layers with `ReLU` activation and a final `Sigmoid` activation layer for binary classification.
- **Overfitting Prevention:**
  - **Dropout Layers:** Added between dense layers to randomly deactivate neurons during training, forcing the network to learn more robust features.
  - **Early Stopping:** The training process was monitored, and `EarlyStopping` was used to halt training when the validation loss stopped improving, ensuring the model with the best performance on unseen data was saved.
- **Performance Optimization:** After training, the model's output probabilities were analyzed to determine an **optimal prediction threshold**. Instead of using the default 0.5, a threshold was chosen to maximize the F1-score, providing a better balance between correctly identifying diabetic and non-diabetic individuals.
- **Model Evaluation:** The final model's performance was assessed using a detailed `classification_report` (providing precision, recall, and F1-score) and a `Confusion Matrix` for a clear visual representation of its predictions versus actual outcomes.
- **Model Persistence:** The trained `MinMaxScaler` object and the final Keras model were saved to disk (`scaler.gz` and `diabetes_model.h5`), allowing for easy loading and reuse in other applications without needing to retrain.

---

## 4. Model Performance

The final, optimized Neural Network achieved a high level of accuracy, typically around **80-85%**. The model demonstrated a strong and balanced ability to predict both classes, as shown by the high F1-scores for both the "Diabetes" and "No Diabetes" categories. The use of regularization and threshold tuning proved effective in creating a reliable and generalized model.

---
