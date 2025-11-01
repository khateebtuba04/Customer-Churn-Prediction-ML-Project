# 📊 Customer Churn Prediction – Beginner-Friendly
# Author: Tuba Mariyam
# Description: Predicting customer churn using Logistic Regression

# ==============================
# 🔧 Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# 📥 Step 1: Load the Dataset
# ==============================
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("✅ Dataset Loaded Successfully!")
print(f"Shape of dataset: {df.shape}")
print("\nFirst 5 rows:\n", df.head())

# ==============================
# 🧹 Step 2: Data Cleaning
# ==============================
# Convert TotalCharges to numeric and drop missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert Yes/No into 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\n✅ Data Cleaned Successfully!")
print("Missing values after cleaning:\n", df.isnull().sum())

# ==============================
# 🔍 Step 3: Basic Data Exploration
# ==============================
sns.set(style="whitegrid", palette="pastel")

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Count (0 = Stayed, 1 = Left)")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Churn by Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.show()

# ==============================
# 🧩 Step 4: Model Building
# ==============================
# Select numeric features for simplicity
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ==============================
# 📈 Step 5: Model Evaluation
# ==============================
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {round(accuracy*100, 2)}%")

print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 💡 Step 6: Final Insights
# ==============================
print("\n📘 Key Insights:")
print("1️⃣ Around 26% of the customers have churned.")
print("2️⃣ Customers with shorter contracts are more likely to leave.")
print("3️⃣ Higher MonthlyCharges lead to a higher chance of churn.")
print("4️⃣ Gender doesn’t strongly influence churn behavior.")

print("\n✅ Project Completed Successfully!")
