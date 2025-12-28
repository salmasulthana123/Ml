# Decision Tree Regression

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------------
# Sample Dataset
# -------------------------------
data = {
    'Area': [1500, 1800, 2400, 3000, 3500, 4000, 600, 1200],
    'Bedrooms': [3, 4, 3, 5, 4, 5, 2, 3],
    'Age': [5, 10, 8, 15, 20, 12, 3, 6],
    'Price': [35, 45, 50, 65, 70, 80, 20, 30]  # price in lakhs
}

df = pd.DataFrame(data)
print("\n=== DATASET ===")
print(df)

# -------------------------------
# Split Data
# -------------------------------
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# -------------------------------
# Decision Tree Regressor
# -------------------------------
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)

pred = tree.predict(X_test)

mse = mean_squared_error(y_test, pred)

print("\nPredicted Prices:", pred)
print("Actual Prices:", y_test.tolist())
print("Mean Squared Error:", mse)
