# Random Forest Classification and Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Sample Dataset
data = {
    "Age": [22, 25, 30, 35, 40, 45, 50, 60],
    "Income": [25000, 30000, 40000, 45000, 60000, 80000, 100000, 120000],
    "Credit_Score": [580, 600, 650, 700, 720, 750, 780, 800],
    "Buy": [0, 0, 1, 1, 1, 1, 1, 1],
    "Spend": [2, 3, 5, 6, 8, 10, 12, 15]
}

df = pd.DataFrame(data)

# Feature Selection
X = df[['Age', 'Income', 'Credit_Score']]
y_class = df['Buy']
y_reg = df['Spend']

# Train/Test split
X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.3)
_, _, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.3)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train_c)
pred_c = clf.predict(X_test)

# Random Forest Regressor
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X_train, y_train_r)
pred_r = reg.predict(X_test)

# Results
print("\nClassification Accuracy:", accuracy_score(y_test_c, pred_c))
print("Predicted Buy Values:", pred_c.tolist())

print("\nRegression MSE:", mean_squared_error(y_test_r, pred_r))
print("Predicted Spend Values:", pred_r.tolist())
