import numpy as np
from scipy import stats
data = [10, 20, 20, 30, 40, 50, 50, 60]
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=True)[0][0]
variance = np.var(data)
std_dev = np.std(data)
range_val = max(data) - min(data) 
print("\n=== DATA ===")
print(data)
print("\n=== CENTRAL TENDENCY ===")
print("Mean =", mean)
print("Median =", median)
print("Mode =", mode)
print("\n=== DISPERSION ===")
print("Variance =", variance)
print("Standard Deviation =", std_dev)
print("Range =", range_val)
