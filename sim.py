import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import time

# Given stock price paths from the paper
stock_paths = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34]
])

# prompt user for option type, strike price, and risk-free rate
type = input("Enter the option type (call/put): ")
strike_price = float(input("Enter the strike price: "))
risk_free_rate = float(input("Enter the risk-free rate: "))

# compute value at expiration
if type == "call":
    option_values = np.maximum(stock_paths[:, -1] - strike_price, 0)
elif type == "put":
    option_values = np.maximum(strike_price - stock_paths[:, -1], 0)
else:
    raise ValueError("Invalid option type")

start_time = time.time()

def discount(values, rate):
    return values * np.exp(-rate)

# Backward induction for optimal stopping
for t in range(stock_paths.shape[1] - 2, 0, -1):  # Going backward from T-1 to 1
    in_the_money = stock_paths[:, t] < strike_price
    X = stock_paths[in_the_money, t].reshape(-1, 1)
    Y = discount(option_values[in_the_money], risk_free_rate)
    
    if len(X) > 0:  # in-the-money options
        X_squared = X ** 2
        X_poly = np.hstack((X, X_squared))

        # ------------------ INSERT MODEL HERE TO CALCULATE CONTINUATION VALUES -------------------
        model = LinearRegression().fit(X_poly, Y)

        # Prediction
        continuation_values = model.predict(X_poly)


        exercise_values = strike_price - X.flatten()
        if type == "call":
            exercise_values = X.flatten() - strike_price
        elif type == "put":
            exercise_values = strike_price - X.flatten()

        exercise_now = exercise_values > continuation_values
        option_values[in_the_money] = np.where(exercise_now, exercise_values, option_values[in_the_money])

# get option value at time 0
option_value = np.mean(discount(option_values, risk_free_rate))

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Estimated American {type.capitalize()} Option Value: {option_value:.4f}")
print(f"Time taken to run the simulation: {elapsed_time:.4f} seconds")