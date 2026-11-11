import numpy as np
import os

class BreastCancerLogisticModel:
    def __init__(self):
        the_path = os.path.dirname(__file__)
        # __file__ → path of the current file (model.py)
        # dirname() → gets the folder containing it

        self.w = np.load(os.path.join(the_path, "model_w.npy"))
        self.b = np.load(os.path.join(the_path, "model_b.npy"))
        self.mu = np.load(os.path.join(the_path, "model_mu.npy"))
        self.sigma = np.load(os.path.join(the_path, "model_sigma.npy"))

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict(self, features):
        x = np.array(features)

        # normalizing
        x = (x - self.mu) / (self.sigma + 1e-10)

        prob = self.sigmoid(np.dot(x, self.w) + self.b)
        # 1 / (1 + exp(-(xw + b)))

        prob_scalar = float(np.squeeze(prob))

        prediction = 1 if prob >= 0.5 else 0
        return prediction, prob_scalar
