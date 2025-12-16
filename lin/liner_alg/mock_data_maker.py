import numpy as np
import pandas as pd
import os

TRUE_W = 2.0
TRUE_B = 10.0

N_SAMPLES = 700

X_MIN = 500
X_MAX = 2500

X_data = np.linspace(X_MIN, X_MAX, N_SAMPLES)

noise = np.random.randn(N_SAMPLES) * 150

Y_data = TRUE_W * X_data + TRUE_B + noise

mock_data = pd.DataFrame({
    'x': X_data,
    'y': Y_data
})

FILE_NAME = 'synthetic_test_data.csv'

mock_data.to_csv(FILE_NAME, index=False)

print(f"Mock Data Generated")
print(f"File saved as: {os.path.abspath(FILE_NAME)}")
print(f"True relationship is Y = {TRUE_W} * X + {TRUE_B}")
print(f"X range: {X_data.min():.0f} to {X_data.max():.0f}")
print(f"Y range: {Y_data.min():.0f} to {Y_data.max():.0f}")