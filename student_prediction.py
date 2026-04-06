# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("Student_Performance.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop student_id if exists
if "student_id" in data.columns:
    data = data.drop("student_id", axis=1)

# Select numeric columns
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Create average score
data["average_score"] = data[numeric_cols].mean(axis=1)

# Create Pass/Fail target
data["pass_fail"] = data["average_score"].apply(lambda x: 1 if x >= 50 else 0)

# Convert categorical columns
data = pd.get_dummies(data)

# Features and target
X = data.drop("pass_fail", axis=1)
y = data["pass_fail"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 SCALE DATA (IMPORTANT for Perceptron)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 🔵 Logistic Regression
# =========================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n🔵 Logistic Regression Accuracy: - student_prediction.py:55", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

plt.figure()
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =================
# 🌳 Decision Tree
# =================
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\n🌳 Decision Tree Accuracy: - student_prediction.py:73", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

plt.figure()
cm_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True, fmt='d')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# ⚡ Perceptron (FIXED)
# =========================
p_model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
p_model.fit(X_train, y_train)
p_pred = p_model.predict(X_test)

print("\n⚡ Perceptron Accuracy: - student_prediction.py:91", accuracy_score(y_test, p_pred))
print(classification_report(y_test, p_pred))

plt.figure()
cm_p = confusion_matrix(y_test, p_pred)
sns.heatmap(cm_p, annot=True, fmt='d')
plt.title("Perceptron Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# 📊 Model Comparison
# =========================
print("\n📊 Model Comparison: - student_prediction.py:105")
print("Logistic Regression: - student_prediction.py:106", accuracy_score(y_test, lr_pred))
print("Decision Tree: - student_prediction.py:107", accuracy_score(y_test, dt_pred))
print("Perceptron: - student_prediction.py:108", accuracy_score(y_test, p_pred))