import os
import numpy as np
import joblib

class BreastCancerSVMModel():
    def __init__(self):
        the_path = os.path.dirname(__file__)
        # __file__ → path of the current file (model.py)
        # dirname() → gets the folder containing it

        self.svm = joblib.load(os.path.join(the_path, 'svm_model.pkl'))
        self.scaler = joblib.load(os.path.join(the_path, 'svm_scaler.pkl'))

    def predict(self, features):
        x = np.array(features)

        # reshape for sklearn
        # Before reshape: (7,)
        # After reshape: (1, 7)
        x = x.reshape(1, -1)

        # normalizing
        x = self.scaler.transform(x)

        prob = self.svm.predict_proba(x)[0][1]
        # inside predict_prob() voting happens and probs are calculated for both
        # classes. structure: [[prob_benign, prob_malignant]]
        # [0][1] → prob_malignant

        prediction = self.svm.predict(x)[0]
        # predict() returns ['class'] so predict()[0] returns 'class'

        prob_scalar = float(np.squeeze(prob))

        return prediction, prob_scalar