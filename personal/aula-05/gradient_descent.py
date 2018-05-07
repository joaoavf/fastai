def gradient_descent(x_arr, y_arr, step, lr, x0=0, x1=0):
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

        # Calculates how much the error changes by the small increment in x1
        dx1 = (err_x1 - mse) / step

        # Sets up new values for x0 and x1
        x0 -= dx0 * lr
        x1 -= dx1 * lr

        # Appends Mean Squared Error to list
        mse_list.append(mse)

    # Calculates the Root Mean Squared Error for the whole cycle
    rmse = sum(mse_list) ** 1 / 2

    # Returns Values
    return x0, x1, rmse

