import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# =========================
# SAMPLE DATASET
# =========================
data = {
    "Hours_Studied": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Sleep_Hours": [9, 8, 7, 7, 6, 5, 5, 4, 3],
    "Score": [50, 55, 60, 65, 70, 78, 85, 90, 95],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print("===== DATASET =====")
print(df)

# =========================
# SPLITTING DATA
# =========================
X = df[['Hours_Studied', 'Sleep_Hours']]
y_class = df['Pass']      # For classification
y_reg = df['Score']       # For regression

X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_class, test_size=0.3
)

_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3
)

# =========================
# KNN CLASSIFICATION
# =========================
knn_cls = KNeighborsClassifier(n_neighbors=3)
knn_cls.fit(X_train, y_train_cls)
pred_cls = knn_cls.predict(X_test)

print("\nClassification Accuracy:",
      accuracy_score(y_test_cls, pred_cls))

# =========================
# KNN REGRESSION
# =========================
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train_reg)
pred_reg = knn_reg.predict(X_test)

print("\nRegression MSE:", mean_squared_error(y_test_reg, pred_reg))
print("\nPredicted Scores:", pred_reg)
