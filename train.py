import pandas as pd
import numpy as np


def normalization(values: list) -> list:
    """Normalizes a list of values to a range between 0 and 1."""
    min_value = min(values)
    max_value = max(values)

    the_list = [((x - min_value) / (max_value - min_value)) for x in values]
    return (the_list)


def main():
    """Trains a simple linear regression model using gradient descent."""
    data = pd.read_csv('data.csv')
    data.columns = data.columns.str.strip()

    y_values = list(data["price"])
    y_values = [float(x) for x in y_values]

    x_values = list(data["km"])
    x_values = [float(x) for x in x_values]

    theta0 = 0
    theta1 = 0

    learning_rate = 0.05

    iterations = 1500

    x_values_norm = normalization(x_values)
    y_values_norm = normalization(y_values)

    m = len(x_values)

    for _ in range(iterations):
        sum_errors0 = 0
        sum_errors1 = 0

        for i in range(m):
            prediction = theta0 + theta1 * x_values_norm[i]
            error = prediction - y_values_norm[i]
            sum_errors0 += error
            sum_errors1 += error * x_values_norm[i]

        theta0 = theta0 - (learning_rate / m) * sum_errors0
        theta1 = theta1 - (learning_rate / m) * sum_errors1
        if np.isnan(theta0) or np.isnan(theta1):
            print("ERROR: theta0 or theta1 became NaN! Stopping training.")
            break

    with open("thetas.txt", "w") as file:
        file.write(
            f"{theta0},{theta1},{min(x_values)}, \
                {max(x_values)},{min(y_values)},{max(y_values)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexcepted error: {e}")
