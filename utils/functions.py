import numpy as np

def calculate_metrics(y_true, y_pred) -> dict[str, float]:

    """
    Compute common regression error metrics.
    takes two same size arrays; predicted vs true values
    Returns a dict with the metric name and value
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}."
        )

    error = y_pred - y_true


    mae = np.mean(np.abs(y_pred - y_true)) #Mean absolute error
    mse = np.mean((y_pred - y_true)**2) #Mean squared error


    # Avoid division by zero for MAPE
    eps = 1e-8
    non_zero_mask = np.abs(y_true) > eps
    if not np.any(non_zero_mask):
        # If everything is zero, MAPE is undefined.
        mape = float("nan")
    else:
        mape = float(
            np.mean(
                np.abs(error[non_zero_mask] / y_true[non_zero_mask])
            ) * 100.0
        )

    return {"mae": mae, "mse": mse, "mape": mape}