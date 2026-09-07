# Linear Regression Model

> **6th Semester University Lab Project**

A collection of supervised machine learning experiments developed for a university laboratory course. The folder contains a linear regression project for medical insurance charge prediction and a classification comparison project using diabetes data.

## Project Contents

### 1. Medical Insurance Premium Prediction

`project.py` uses Linear Regression to estimate an individual's medical insurance charge from demographic and lifestyle information.

The model uses:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The script encodes categorical data, splits the dataset into training and testing sets, trains a Linear Regression model, reports evaluation metrics, displays an actual-vs-predicted graph, and accepts a custom user profile for prediction.

### 2. Diabetes Classification Comparison

`hw.py` compares three classification algorithms using `diabetes.csv`:

- Random Forest Classifier
- Decision Tree Classifier
- K-Nearest Neighbors Classifier

Each model is evaluated using accuracy, precision, recall, F1 score, and a confusion matrix. The script identifies the model with the highest test accuracy.

## Datasets

- `insurance.csv`: Medical insurance records with `1338` observations.
- `diabetes.csv`: Diabetes diagnostic data with an `Outcome` target column.

## Requirements

- Python 3.10+
- pandas
- NumPy
- scikit-learn
- matplotlib

Install the dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## How to Run

Open a terminal in this folder:

```bash
cd "Linear Regression Model"
```

Run the insurance charge prediction project:

```bash
python project.py
```

Run the diabetes classification comparison:

```bash
python hw.py
```

## Insurance Prediction Input Example

When `project.py` asks for input, a valid example is:

```text
Age: 25
Sex: male
BMI: 25.5
Children: 0
Smoker: no
Region: northeast
```

The script prints the predicted insurance charge and displays a scatter plot comparing actual and predicted charges.

## Evaluation Metrics

The projects use standard regression and classification metrics:

- Mean Squared Error (MSE)
- R² score
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

## Academic Note

This folder was developed as part of the **6th Semester University Laboratory Projects**. The models are intended for educational experimentation and should not be treated as professional medical, financial, or clinical decision-making systems.
