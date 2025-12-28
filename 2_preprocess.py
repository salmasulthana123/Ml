import pandas as pd
import numpy as np
data = {'Age': [18, 22, 25, 120, np.nan, 30, 29, 100],'Marks': [80, 85, np.nan, 75, 95, 92, 88, 91],'Dept': ["CSE", "ECE", "ECE", "CSE", "IT", "CSE", np.nan, "ECE"]
}
df = pd.DataFrame(data)
print("Original Data:\n", df)
df['Marks'].fillna(df['Marks'].mean(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Dept'].fillna(df['Dept'].mode()[0], inplace=True)
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Age'] >= Q1 - 1.5 * IQR) & (df['Age'] <= Q3 + 1.5 * IQR)]
df_selected = df[['Age', 'Marks']]
df_selected['Marks_Category'] = pd.cut(df_selected['Marks'],_bins=[0, 70, 85, 100],labels=["Low", "Medium", "High"])
print("\nAfter Preprocessing:\n", df_selected)
