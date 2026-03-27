# ============================================================
# CUSTOMER CHURN PREDICTION (EDA + MACHINE LEARNING)
# ============================================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded:", df.shape)
print(df.head())

# ============================================================
# STEP 3: DATA CLEANING
# ============================================================

# Convert TotalCharges to numeric (fix hidden strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values
df.dropna(inplace=True)

# Drop customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# ============================================================
# STEP 4: EDA (VISUALIZATION)
# ============================================================

# Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.savefig("churn_distribution.png")
plt.show()

# Churn vs Contract
plt.figure(figsize=(6,4))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.savefig("churn_contract.png")
plt.show()

# Monthly Charges distribution
plt.figure(figsize=(8,5))
sns.histplot(df['MonthlyCharges'], bins=30, kde=True)
plt.title("Monthly Charges Distribution")
plt.savefig("monthly_charges.png")
plt.show()

# ============================================================
# STEP 5: ENCODING (FIXED VERSION)
# ============================================================

# Convert categorical columns to numeric using One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# ============================================================
# STEP 6: MODEL BUILDING
# ============================================================

# Features and target
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ============================================================
# STEP 7: EVALUATION
# ============================================================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))