def adam(x_arr, y_arr, step, lr, beta, beta2, x0=0, x1=0, momentum_x0=0, momentum_x1=0, sq_momentum_x0=0,
         sq_momentum_x1=0):
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

        # Momentum X0
        momentum_x0 = dx0 * (1 - beta) + momentum_x0 * beta

        # Calculates how much the error changes by the small increment in x1
        dx1 = (err_x1 - mse) / step

        # Momentum X1
        momentum_x1 = dx1 * (1 - beta) + momentum_x1 * beta

        # Squared Momentum X0
        sq_momentum_x0 = momentum_x0 ** 2 * (1 - beta2) + dx0 ** 2 * beta2

        # Squared Momentum X0
        sq_momentum_x1 = momentum_x1 ** 2 * (1 - beta2) + dx1 ** 2 * beta2

        # Sets up new values for x0 and x1
        x0 -= momentum_x0 * lr / (sq_momentum_x0 ** 1 / 2)
        x1 -= momentum_x1 * lr / (sq_momentum_x1 ** 1 / 2)

        # Appends Mean Squared Error to list
        mse_list.append(mse)

    # Calculates the Root Mean Squared Error for the whole cycle
    rmse = sum(mse_list) ** 1 / 2

    # Returns Values
    return x0, x1, momentum_x0, momentum_x1, sq_momentum_x0, sq_momentum_x1, rmse


import numpy as np

x_100 = np.random.randint(1050, size=100)
y_100 = x_100 * 2 + 200

x0, x1, momentum_x0, momentum_x1 = 0, 0, 0, 0
rmse_list = []

x0, x1, momentum_x0, momentum_x1, sq_momentum_x0, sq_momentum_x1 = 0, 0, 0, 0, 0, 0

for w in range(0, 1000):
    x0, x1, momentum_x0, momentum_x1, sq_momentum_x0, sq_momentum_x1, rmse = adam(x_100, y_100, 0.00001, 100, .7, .9,
                                                                                  x0, x1, momentum_x0, momentum_x1,
                                                                                  sq_momentum_x0, sq_momentum_x1)
rmse_list.append(rmse)
