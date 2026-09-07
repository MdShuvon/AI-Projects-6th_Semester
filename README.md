# AI Projects - 6th Semester University Lab

A curated collection of machine learning and data science projects jointly developed with classmates as part of **6th Semester University Laboratory Projects**.

The repository demonstrates practical applications of supervised learning, feature engineering, model evaluation, sports analytics, regression, and classification.

## Projects

| Project | Focus | Main Techniques |
| --- | --- | --- |
| [FIFA World Cup Winner Prediction](FIFA%20World%20Cup%20Winner%20Prediction/README.md) | Football match and tournament prediction | Elo rating, Gradient Boosting, recent-form statistics, probability-based simulation |
| [Linear Regression Model](Linear%20Regression%20Model/README.md) | Insurance charge prediction and diabetes classification | Linear Regression, Random Forest, Decision Tree, KNN |

## Repository Structure

```text
AI Projects - 6th Semester/
├── FIFA World Cup Winner Prediction/
│   ├── FIFA.py
│   ├── results.csv
│   └── README.md
├── Linear Regression Model/
│   ├── project.py
│   ├── hw.py
│   ├── insurance.csv
│   ├── diabetes.csv
│   └── README.md
└── README.md
```

## Technologies

- Python 3.10+
- pandas
- NumPy
- scikit-learn
- matplotlib
- Git and GitHub

## Environment Setup

Create and activate a virtual environment from the repository root:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Running the Projects

### FIFA World Cup Prediction

From the repository root:

```bash
python "FIFA World Cup Winner Prediction/FIFA.py"
```

The program trains a classifier from historical international match results, calculates team strength, simulates a 48-team tournament, and provides a custom match prediction.

### Medical Insurance Prediction

```bash
cd "Linear Regression Model"
python project.py
```

This program trains a Linear Regression model, evaluates predicted insurance charges, displays an actual-vs-predicted graph, and accepts a custom profile.

### Diabetes Classification Comparison

From inside `Linear Regression Model`:

```bash
python hw.py
```

This program compares Random Forest, Decision Tree, and KNN classifiers using accuracy, precision, recall, F1 score, and confusion matrices.

## Learning Outcomes

These laboratory projects provide hands-on practice with:

- Loading and cleaning real-world datasets
- Encoding categorical variables
- Splitting data into training and testing sets
- Training multiple supervised learning models
- Comparing models with standard evaluation metrics
- Building feature-based prediction workflows
- Presenting machine learning results through console output and visualizations

## Academic Disclaimer

This repository is a collaborative academic submission and learning portfolio created with classmates for **6th Semester University Laboratory Projects**. The predictions and model outputs are intended for educational experimentation only. They are not official sports forecasts, medical advice, insurance quotations, or professional decision-making tools.

Repository: [AI-Projects-6th_Semester](https://github.com/MdShuvon/AI-Projects-6th_Semester)
