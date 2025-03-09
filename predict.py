import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def r2_manual(y_true, y_pred):
    """Computes the R-squared (R²) score as a percentage."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    r2 = 1 - (ss_residual / ss_total)
    return r2 * 100


def plot_graph(theta0, theta1, min_x, max_x, min_y, max_y):
    """Plots a linear regression model with actual data points."""
    data = pd.read_csv("data.csv")
    x_values = list(data["km"])
    y_values = list(data["price"])

    if max_x == min_x:
        print("Error: max_x and min_x are the same. Cannot normalize.")
        return
    normalized_x_values = [(x - min_x) / (max_x - min_x) for x in x_values]

    predicted_prices_norm = [theta0 + theta1 * x for x in normalized_x_values]

    predicted_prices = [y * (max_y - min_y) +
                        min_y for y in predicted_prices_norm]

    accurancy = r2_manual(y_values, predicted_prices)
    print(f"Accuracy: {accurancy:.2f}%")

    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title("Mileage-Price Regression")
    plt.scatter(x_values, y_values, color="blue", label="Actual Data")
    plt.plot(x_values, predicted_prices, color="red", label="Regression Line")

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Linear Regression Model")
    plt.legend()

    x_range = max_x - min_x
    y_range = max_y - min_y
    plt.xlim(min_x - 0.1 * x_range, max_x + 0.1 * x_range)
    plt.ylim(min_y - 0.1 * y_range, max_y + 0.1 * y_range)
    plt.show()


def main():
    """Predicts the price of a car based on mileage
      using a trained linear regression model."""
    theta0 = 0
    theta1 = 0
    try:
        km_input = float(input("Enter the mileage (km): "))
    except ValueError:
        print("Error: Please enter a valid number for mileage.")
        return

    try:
        with open("thetas.txt", "r") as file:
            content = file.read().strip()
        if not content:
            print("Model hasn't been trained yet. \
                Returning 0 as default prediction.")
            print(f"Estimated price for {km_input:.2f} km: 0.00 €")
            return

        parts = content.split(",")
        if len(parts) < 6:
            print("Error: The thetas.txt file does not contain enough data.")
            return

        theta0 = float(parts[0])
        theta1 = float(parts[1])
        min_x = float(parts[2])
        max_x = float(parts[3])
        min_y = float(parts[4])
        max_y = float(parts[5])

    except FileNotFoundError:
        print("Model hasn't been trained yet.")
        print(f"Estimated price for {km_input:.2f} km: 0.00 €")
        return
    except ValueError:
        print("Error: Invalid value found in thetas.txt.")
        return

    if max_x == min_x:
        print("Error: max_x and min_x are the same. Cannot normalize.")
        return
    normalized_km = (km_input - min_x) / (max_x - min_x)

    predicted_y_norm = theta0 + theta1 * normalized_km

    if max_y == min_y:
        print("Error: max_y and min_y are the same. Cannot denormalize.")
        return
    predicted_y = predicted_y_norm * (max_y - min_y) + min_y

    print(f"Estimated price: {predicted_y:.2f} €")

    plot_graph(theta0, theta1, min_x, max_x, min_y, max_y)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexcepted error: {e}")
