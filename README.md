# Diabetes Prediction Model

## Overview

The **Diabetes Prediction Model** is a machine learning model that predicts the likelihood of a patient being diabetic based on various health parameters. This model uses **logistic regression** as its primary algorithm due to its effectiveness in binary classification problems. The project aims to provide an accessible way for healthcare providers and individuals to gain early insights into potential diabetic conditions, which can help in early diagnosis and intervention.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Model Training](#model-training)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [License](#license)

## Introduction

Diabetes is a chronic health condition that affects millions of people worldwide. Early detection and intervention can help in managing and even preventing complications associated with diabetes. This project utilizes a logistic regression model to predict the likelihood of diabetes based on a dataset of patients' health information.

## Dataset

The dataset used for this project is the **Pima Indians Diabetes Dataset**, which is commonly used in machine learning for diabetes prediction. It includes the following features:

1. Number of Pregnancies
2. Glucose Level
3. Blood Pressure
4. Skin Thickness
5. Insulin Level
6. Body Mass Index (BMI)
7. Diabetes Pedigree Function
8. Age

The target variable is binary, indicating whether the patient is diabetic (1) or not (0).

## Features

The primary features for prediction in this model include:

- **Pregnancies**: Number of times the patient has been pregnant.
- **Glucose**: Plasma glucose concentration (mg/dL).
- **Blood Pressure**: Diastolic blood pressure (mm Hg).
- **Skin Thickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index, calculated as weight in kg/(height in m)^2.
- **Diabetes Pedigree Function**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Patient’s age.

## Model Training

The logistic regression model was chosen for this project due to its suitability for binary classification. The model was trained using **scikit-learn's LogisticRegression** implementation, which is well-suited for healthcare applications where interpretability is key. 

Steps:

1. **Data Preprocessing**: Handled missing values, normalized the dataset, and split the data into training and test sets.
2. **Model Training**: Applied logistic regression with regularization to avoid overfitting.
3. **Model Evaluation**: Evaluated the model’s accuracy, precision, recall, and F1-score on the test dataset.

## Requirements

- Python 3.7 or later
- pandas
- numpy
- scikit-learn
- matplotlib (optional, for visualizations)

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Diabetes-Prediction-Model.git
   ```

2. Change directory into the project folder:

   ```bash
   cd Diabetes-Prediction-Model
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the Model**: To train the model with the dataset, run the following command:

   ```bash
   python train_model.py
   ```

   This will train the model and save it to a file (e.g., `diabetes_model.pkl`).

2. **Make Predictions**: To make predictions on new data, run:

   ```bash
   python predict.py --input_data your_input_data.csv
   ```

3. **Evaluate the Model**: You can evaluate the model on a test dataset by running:

   ```bash
   python evaluate_model.py
   ```

## Evaluation

The model's performance was assessed based on various metrics:

- **Accuracy**: Measures the proportion of correct predictions.
- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the ability of the model to find all positive samples.
- **F1 Score**: Harmonic mean of precision and recall, providing a balanced evaluation metric.

The evaluation results are stored in `evaluation_results.txt` and visualized in `evaluation_plots/`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
