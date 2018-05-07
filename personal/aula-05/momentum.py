def momentum(x_arr, y_arr, step, lr, beta, x0=0, x1=0, momentum_x0=0, momentum_x1=0):
    # Initializes Mean Squared Error List
    mse_list = []
    for x, y in zip(x_arr, y_arr):
        # Make prediction for a value of x
        prediction = x0 + x1 * x

        # Calculates Mean Squared Error
        mse = (y - prediction) ** 2

        # Calculates error for small increments in x0
        err_x0 = ((x0 + step + x1 * x) - y) ** 2

        # Calculates error for small increments in x1
        err_x1 = ((x0 + (x1 + step) * x) - y) ** 2

        # Calculates how much the error changes by the small increment in x0
        dx0 = (err_x0 - mse) / step

        momentum_x0 = dx0 * (1 - beta) + momentum_x0 * beta

        # Calculates how much the error changes by the small increment in x1
        dx1 = (err_x1 - mse) / step

        momentum_x1 = dx1 * (1 - beta) + momentum_x1 * beta

        # Sets up new values for x0 and x1
        x0 -= momentum_x0 * lr
        x1 -= momentum_x1 * lr

        # Appends Mean Squared Error to list
        mse_list.append(mse)

    # Calculates the Root Mean Squared Error for the whole cycle
    rmse = sum(mse_list) ** 1 / 2

    # Returns Values
    return x0, x1, momentum_x0, momentum_x1, rmse

import numpy as np

x_100 = np.random.randint(1050, size=100)
y_100 = x_100 * 2 + 200

x0, x1, momentum_x0, momentum_x1 = 0, 0, 0, 0
rmse_list = []
for w in range(0, 1000):
    x0, x1, momentum_x0, momentum_x1, rmse = momentum(x_100, y_100, 0.00001, 1/10**6, .9, x0, x1, momentum_x0, momentum_x1)
    rmse_list.append(rmse)
    print(rmse)