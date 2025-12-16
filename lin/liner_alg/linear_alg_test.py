import numpy as np 
import pandas as pd

user_input = r'./data_sets/test_data.csv'

try: 
    data_set = pd.read_csv(user_input)
except FileNotFoundError:
    print(f'File not found')

X = data_set['x'].values # returning npdarray
Y = data_set['y'].values

    
def model(w, X, b): # model (f_wb)
    # X - numpy array 
    return w*X + b 

    # w is multiplied by every single element in the X array
    # b is then added to every single element of the result of w * x
    # f_wb is an array of 700 elements, where each element is prediction for the corresponding x value.


def cost_function(X, Y, w, b):
    # X (ndarray) - input feature vector
    # Y (ndarray) - Target value vector
    # w (float) - weight parameter
    # b (float) - bias parameter

    m = X.shape[0] # number of training examples

    f_wb = model(X, w, b) # calculate the predictions

    error = f_wb - Y # calculating the error

    total_cost = (1 / (2 * m) * np.sum(error**2))

    return total_cost


def compute_gradient(X, Y, w, b):
    # calculating the average slope across all data points
    m = X.shape[0]

    f_wb = model(w, X, b)
    
    error = f_wb - Y

    dj_db = np.sum(error) / m # sort of an everage of error through out our dataset

    dj_dw = np.sum(error * X) / m # the same but with x

    return dj_dw, dj_db

def gradient_descend(X, Y, w_in = 30, b_in = 30, alpha = 0.0001, epochs = 20000):
    w = w_in
    b = b_in
    J_history = [] # monitoring convergence

    print(f"Starting Gradient Descent with alpha={alpha}, iterations={epochs}")

    for i in range(epochs):
        dj_dw, dj_db = compute_gradient(X, Y, w, b) # calculating the slopes

        # updating parameters simultaneously

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = cost_function(X, Y, w, b)
        J_history.append(cost)

        if i % (epochs // 10) == 0:
            print(f"Iteration {i:5d} | Cost: {cost:.4f} | w: {w:.4f} | b: {b:.4f}")

    return w, b, J_history

w, b, J_history = gradient_descend(X, Y)

    
# Prediction 

print('\nmaking prediction')

x_predict_single = 65.0

y_predict_single = model(w, x_predict_single, b)

print(f"Final Parameters: w={w:.4f}, b={b:.4f}")
print(f"Prediction for X = {x_predict_single:.1f}: Y_hat = {y_predict_single:.4f}")

# predicting a vector of new values 

X_new = np.array([10.0, 50.0, 95.0, 110.0]) 
y_predictions_vector = model(w, X_new, b)

print(f"\nPredictions for new data batch:")
for x_val, y_hat in zip(X_new, y_predictions_vector):
    print(f" X = {x_val:.1f} -> Y_hat = {y_hat:.4f}")

