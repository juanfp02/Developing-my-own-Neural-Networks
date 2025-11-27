import numpy as np

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true)**2)
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    return {"mae": mae, "mse": mse, "mape": mape}