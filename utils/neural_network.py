import numpy as np


# ===========================
# Embedding Layer (NumPy)
# ===========================
class Embedding:
    def __init__(self, num_categories, embed_dim):
        self.num_categories = num_categories
        self.embed_dim = embed_dim
        self.weights = 0.01 * np.random.randn(num_categories, embed_dim)
        self.dweights = np.zeros_like(self.weights)
        self.inputs = None

    def forward(self, inputs):
        # inputs: integer indices
        inputs = inputs.astype(int)
        self.inputs = inputs

        max_idx = inputs.max()
        if max_idx >= self.weights.shape[0]:
            # Auto-expand embedding matrix to handle unexpected indices
            extra = max_idx + 1 - self.weights.shape[0]
            new_rows = 0.01 * np.random.randn(extra, self.embed_dim)
            self.weights = np.vstack([self.weights, new_rows])
            self.dweights = np.zeros_like(self.weights)

        return self.weights[self.inputs]

    def backward(self, dvalues):
        self.dweights = np.zeros_like(self.weights)
        for i, idx in enumerate(self.inputs):
            self.dweights[idx] += dvalues[i]


# ===========================
# Dense Layer
# ===========================
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_reg_l2=0.0, bias_reg_l2=0.0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_reg_l2 = weight_reg_l2
        self.bias_reg_l2 = bias_reg_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases

    def backward(self, dvalues):
        # Gradients
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L2 regularization
        if self.weight_reg_l2 > 0:
            self.dweights += 2 * self.weight_reg_l2 * self.weights
        if self.bias_reg_l2 > 0:
            self.dbiases += 2 * self.bias_reg_l2 * self.biases

        # Gradient wrt inputs
        self.dinputs = dvalues @ self.weights.T


# ===========================
# ReLU Activation
# ===========================
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# ===========================
# MSE Loss
# ===========================
class Loss_MeanSquaredError:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        self.dinputs = 2 * (y_pred - y_true) / samples

# ===========================
# Full Model Wrapper (optional)
# ===========================
# You may or may not use this; forward pass controlled in main.