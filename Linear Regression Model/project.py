# ================================================
# Project: Medical Insurance Premium Prediction
# Model  : Linear Regression
# Dataset: insurance.csv (1338 rows)
# ================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ── 1. Load Dataset ──────────────────────────────
df = pd.read_csv("insurance.csv")
print("✅ Dataset Loaded! Shape:", df.shape)
print(df.head())

# ── 2. Encode Text Columns ───────────────────────
df['sex']    = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)
print("\n✅ Encoding done! Columns:", df.columns.tolist())

# ── 3. Features & Target ─────────────────────────
X = df.drop('charges', axis=1)
y = df['charges']

# ── 4. Train / Test Split ────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\n✅ Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── 5. Train the Model ───────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model trained!")

# ── 6. Evaluate ──────────────────────────────────
y_pred = model.predict(X_test)
mse    = mean_squared_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
print("\n════════ Model Results ════════")
print(f"MSE      : {mse:,.2f}")
print(f"R² Score : {r2:.4f}")
print(f"Accuracy : {r2*100:.1f}%")

# ── 7. Graph: Actual vs Predicted ────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue', label='Predictions')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect fit')
plt.xlabel("Actual Charges ($)")
plt.ylabel("Predicted Charges ($)")
plt.title("Actual vs Predicted Insurance Charges")
plt.legend()
plt.tight_layout()
plt.show()

# ── 8. User Input → Prediction ───────────────────
print("\n" + "═"*44)
print("   Insurance Premium Predictor")
print("═"*44)

age      = float(input("Age (e.g. 25): "))
sex_in   = input("Sex (male/female): ").strip().lower()
sex      = 1 if sex_in == 'male' else 0
bmi      = float(input("BMI (e.g. 25.5): "))
children = int(input("Children (0-5): "))
smk_in   = input("Smoker (yes/no): ").strip().lower()
smoker   = 1 if smk_in == 'yes' else 0
region   = input("Region (northeast/northwest/southeast/southwest): ").strip().lower()

r_nw = 1 if region == 'northwest' else 0
r_se = 1 if region == 'southeast'  else 0
r_sw = 1 if region == 'southwest'  else 0

user_df = pd.DataFrame({
    'age'              : [age],
    'sex'              : [sex],
    'bmi'              : [bmi],
    'children'         : [children],
    'smoker'           : [smoker],
    'region_northwest' : [r_nw],
    'region_southeast' : [r_se],
    'region_southwest' : [r_sw]
})

predicted = model.predict(user_df)[0]
print(f"\n  ✅ Predicted Insurance Charge: ${predicted:,.2f}")
print("═"*44)
