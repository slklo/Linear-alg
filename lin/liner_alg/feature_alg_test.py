import numpy as np
import pandas as pd 

# data_set load

try: 
    data_set = pd.read_csv('./data_sets/synthetic_test_data.csv')
except FileNotFoundError: 
    print('File not found')

X = data_set['x'].values
Y = data_set['y'].values

# Z-score normalization  

mu = np.mean(X)
sigma = np.std(X)

X_norm = (X - mu) / sigma


# Functionality

def model(w, X, b):
    return w*X + b

def cost_function(X, Y, w, b):
    m = X.shape[0]

    f_wb = model(w, X, b)

    return (1 / (2 * m) * np.sum((f_wb - Y) ** 2))


def compute_gradient(X, Y, w, b):

    m = X.shape[0]
    f_wb = model(w, X, b)
    dj_db = np.sum((f_wb - Y)) / m
    dj_dw = np.sum((f_wb - Y) * X) / m

    return dj_dw, dj_db

def gradient_descend(X, Y, w_in = 0, b_in = 0, alpha = 0.01, epochs= 10000):
    w = w_in
    b = b_in
    J_history = []

    for i in range(epochs):
        dj_dw, dj_db = compute_gradient(X, Y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = cost_function(X, Y, w, b)

        J_history.append(cost)

        if i % (epochs // 10) == 0:
            print(f"Iteration {i:5d} | Cost: {cost:.4f} | w: {w:.4f} | b: {b:.4f}")

    return w, b, J_history

w, b, J_history = gradient_descend(X_norm, Y, alpha=0.5, epochs=1000)


# Predictions

print('\n Predictions')

final_w = w
final_b = b

x_predict_single = 65.0

x_predict_single_norm = (x_predict_single - mu) / sigma 

y_predict_single = model(final_w, x_predict_single_norm, final_b)

print(f"Final Parameters (Scaled W, B): w={final_w:.4f}, b={final_b:.4f}")
print(f"Prediction for X = {x_predict_single:.1f}: Y_hat = {y_predict_single:.4f}")

X_new = np.array([500.0, 600.0, 700.0, 800.0]) 
X_new_norm = (X_new - mu) / sigma 

y_predictions_vector = model(final_w, X_new_norm, final_b)

print(f"\nPredictions for new data batch (Corrected):")
for x_val, y_hat in zip(X_new, y_predictions_vector):
    print(f" X = {x_val:.1f} -> Y_hat = {y_hat:.4f}")