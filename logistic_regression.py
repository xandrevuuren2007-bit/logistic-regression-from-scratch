import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("C:/Users/xandr/Desktop/Xandre/Programming/Python projects/penguins.csv")
df = df[[
    "bill_length_mm", 
    "bill_depth_mm", 
    "flipper_length_mm",
    "sex"
    ]].dropna()

# Features
X = df[[
    "bill_length_mm" , 
    "bill_depth_mm", 
    "flipper_length_mm"
    ]].to_numpy()

# Standardize features
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

# Target
y = df["sex"].to_numpy()

# Model params
w = np.zeros(X.shape[1])
b = 0

#Hyper-params
learning_rate = 0.1
threshold = 0.5
max_iterations = 1000

# Training utils
n = X.shape[0]
epsilon = 1e-15
loss_history = []

binary = []

for sex in y:
    if sex == "male":
        b_val = 1
    else:
        b_val = 0
    binary.append(b_val)
y = np.array(binary)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient descent

for i in range(max_iterations):

    z = X @ w + b
    p = sigmoid(z)
    p_safe = np.clip(p, epsilon, 1 - epsilon)

    current_loss = -np.mean(
        y * np.log(p_safe)
        + (1 - y) * np.log(1 - p_safe))
    loss_history.append(current_loss)

    error = p - y
    dw = (1 / n) * (X.T @ error)
    db = np.mean(error)

    w -= learning_rate * dw
    b -= learning_rate * db

# Final predictions
p = sigmoid(X @ w + b)
predictions = (p >= threshold).astype(int)  

# Evaluation
accuracy = np.mean(predictions == y) * 100
print(f"Actual values: {y}")
print(f"Prediction's values: {predictions}")
print(f"\nInitial loss: {loss_history[0]:.6f}")
print(f"Final loss: {loss_history[-1]:.6f}")
print(f"\n🎉 SUCCESS! Training Accuracy: {accuracy:.2f}%")
