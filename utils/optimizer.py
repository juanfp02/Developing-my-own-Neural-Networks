import numpy as np

class Optimizer_Adam:
    """
    Adam optimizer for layers
    """



    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        """Update weights (and biases, if present) of a single layer"""

        # If the layer does not have trainable weights, skip it
        if not hasattr(layer, "weights"):
            return

        # Lazily initialize moment estimates on first update
        if not hasattr(layer, "weight_m"):
            layer.weight_m = np.zeros_like(layer.weights)
            layer.weight_v = np.zeros_like(layer.weights)
        if hasattr(layer, "biases") and not hasattr(layer, "bias_m"):
            layer.bias_m = np.zeros_like(layer.biases)
            layer.bias_v = np.zeros_like(layer.biases)


        # --- Weights ---
        # Exponential moving averages of the gradients (first moment)
        layer.weight_m = self.beta_1 * layer.weight_m + (1 - self.beta_1) * layer.dweights
        # Exponential moving averages of squared gradients (second moment)
        layer.weight_v = self.beta_2 * layer.weight_v + (1 - self.beta_2) * (layer.dweights ** 2)

        # Bias correction to counteract initialization at zero
        m_hat_w = layer.weight_m / (1 - self.beta_1 ** (self.iterations + 1))
        v_hat_w = layer.weight_v / (1 - self.beta_2 ** (self.iterations + 1))

        # Parameter update
        layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

        # --- Biases ---
        if hasattr(layer, "biases"):
            layer.bias_m = self.beta_1 * layer.bias_m + (1 - self.beta_1) * layer.dbiases
            layer.bias_v = self.beta_2 * layer.bias_v + (1 - self.beta_2) * (layer.dbiases ** 2)

            m_hat_b = layer.bias_m / (1 - self.beta_1 ** (self.iterations + 1))
            v_hat_b = layer.bias_v / (1 - self.beta_2 ** (self.iterations + 1))

            layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1