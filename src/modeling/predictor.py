import numpy as np
import joblib

class Predictor:
    def __init__(self, save_model_path: str):
        self.model = joblib.load(save_model_path)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred_col = self.model.predict(X_test)
        return y_pred_col