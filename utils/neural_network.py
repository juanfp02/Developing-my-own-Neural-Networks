import numpy as np

"""
Core neural network components implemented with NumPy.

This module defines:
- Embedding layer for categorical features.
- Fully-connected (dense) layers with optional L2 regularization.
- ReLU activation.
- Mean squared error loss.

These building blocks are used by the training script (see main.ipynb)
to construct and train a feed-forward regression model.
"""

# ===========================
# Embedding Layer (NumPy)
# ===========================
class Embedding:
    def __init__(self, num_categories, embed_dim):
        # Number of possible categorical values (row count of embedding table)
        self.num_categories = num_categories
        # Dimensionality of each embedding vector
        self.embed_dim = embed_dim

        self.weights = 0.01 * np.random.randn(num_categories, embed_dim)
        self.dweights = np.zeros_like(self.weights)
        self.inputs = None

    def forward(self, inputs):
        """
        Forward pass: takes integer category indices and returns
        their corresponding embedding vectors.
        """
        inputs = inputs.astype(int)
        self.inputs = inputs

        # Find the largest index present in this batch
        max_idx = inputs.max()

        # If an index falls outside the current table, grow the table
        if max_idx >= self.weights.shape[0]:
            extra = max_idx + 1 - self.weights.shape[0]
            new_rows = 0.01 * np.random.randn(extra, self.embed_dim)
            self.weights = np.vstack([self.weights, new_rows])
            self.dweights = np.zeros_like(self.weights)

        # Return the embedding vectors for the given indices
        return self.weights[self.inputs]

    def backward(self, dvalues):
        """
        Backward pass: accumulates gradients for each embedding index.
        """
        # Reset gradients
        self.dweights = np.zeros_like(self.weights)

        # Accumulate gradients for each index in the batch
        for i, idx in enumerate(self.inputs):
            self.dweights[idx] += dvalues[i]


# ===========================
# Dense Layer
# ===========================
class Layer_Dense:
    """
    Fully-connected (dense) layer.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    n_neurons : int
        Number of neurons (output units).
    weight_reg_l2 : float, optional
        L2 regularization strength for weights.
    bias_reg_l2 : float, optional
        L2 regularization strength for biases.
    """


    def __init__(self, n_inputs, n_neurons, weight_reg_l2=0.0, bias_reg_l2=0.0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_reg_l2 = weight_reg_l2
        self.bias_reg_l2 = bias_reg_l2

    def forward(self, inputs):

        """
        Compute layer outputs: X @ W + b.
        """

        
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Expected inputs with {self.weights.shape[0]} features, "
                f"got {inputs.shape[1]}."
            )

        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases

    def backward(self, dvalues):
        """
        Backward pass for dense layer.
        """

        # Gradient with respect to weights
        self.dweights = self.inputs.T @ dvalues

        # Add L2 regularization gradient if enabled
        if self.weight_reg_l2 > 0:
            self.dweights += 2 * self.weight_reg_l2 * self.weights

        # Gradient with respect to biases (sum across batch)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.bias_reg_l2 > 0:
            self.dbiases += 2 * self.bias_reg_l2 * self.biases

        # Gradient with respect to inputs
        self.dinputs = dvalues @ self.weights.T


# ===========================
# ReLU Activation
# ===========================
class Activation_ReLU:

    """ReLU activation: f(x) = max(0, x)."""

    def __init__(self) -> None:
        self.inputs: np.ndarray | None = None
        self.output: np.ndarray | None = None
        self.dinputs: np.ndarray | None = None


    def forward(self, inputs):
        """Apply ReLU activation elementwise."""
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """Apply ReLU activation elementwise."""
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# ===========================
# MSE Loss
# ===========================
class Loss_MeanSquaredError:
    """Mean squared error (MSE) loss for regression."""
    def forward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape.")
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        self.dinputs = 2 * (y_pred - y_true) / samples