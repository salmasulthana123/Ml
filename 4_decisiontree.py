# Decision Tree Classification with Parameter Tuning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Sample Dataset
# -------------------------------
data = {
    "Age": [25, 30, 45, 35, 22, 40, 28, 48],
    "Income": ["High", "Medium", "High", "Medium", "Low", "Low", "Medium", "High"],
    "Student": ["No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No"],
    "Buys_Computer": ["No", "Yes", "Yes", "Yes", "No", "No", "Yes", "Yes"]
}

df = pd.DataFrame(data)

# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop("Buys_Computer_Yes", axis=1)
y = df["Buys_Computer_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# -------------------------------
# Base Model (No Tuning)
# -------------------------------
dt1 = DecisionTreeClassifier()
dt1.fit(X_train, y_train)
pred1 = dt1.predict(X_test)

# -------------------------------
# Tuned Model
# -------------------------------
dt2 = DecisionTreeClassifier(max_depth=3, criterion="entropy")
dt2.fit(X_train, y_train)
pred2 = dt2.predict(X_test)

print("Accuracy (Without Tuning):", accuracy_score(y_test, pred1))
print("Accuracy (With Tuning):", accuracy_score(y_test, pred2))
