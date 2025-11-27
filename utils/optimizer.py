import numpy as np

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        # Embedding layer OR Dense layer (both have weights)
        if not hasattr(layer, "weights"):
            return

        if not hasattr(layer, "weight_cache"):
            layer.weight_m = np.zeros_like(layer.weights)
            layer.weight_v = np.zeros_like(layer.weights)
            layer.bias_m = np.zeros_like(layer.biases) if hasattr(layer, "biases") else None
            layer.bias_v = np.zeros_like(layer.biases) if hasattr(layer, "biases") else None

        # --- Weights ---
        layer.weight_m = self.beta_1 * layer.weight_m + (1 - self.beta_1) * layer.dweights
        layer.weight_v = self.beta_2 * layer.weight_v + (1 - self.beta_2) * (layer.dweights ** 2)

        m_hat_w = layer.weight_m / (1 - self.beta_1 ** (self.iterations + 1))
        v_hat_w = layer.weight_v / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

        # --- Biases (Dense layers only) ---
        if hasattr(layer, "biases"):
            layer.bias_m = self.beta_1 * layer.bias_m + (1 - self.beta_1) * layer.dbiases
            layer.bias_v = self.beta_2 * layer.bias_v + (1 - self.beta_2) * (layer.dbiases ** 2)

            m_hat_b = layer.bias_m / (1 - self.beta_1 ** (self.iterations + 1))
            v_hat_b = layer.bias_v / (1 - self.beta_2 ** (self.iterations + 1))

            layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1