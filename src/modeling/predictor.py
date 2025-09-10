import numpy as np
import joblib
import os
import glob
import logging
import xgboost as xgb

class Predictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = self._load_models()

    def _load_models(self):
        """从指定目录加载所有模型文件。"""
        model_paths = sorted(glob.glob(os.path.join(self.model_dir, 'model_target_*.joblib')))
        if not model_paths:
            raise FileNotFoundError(f"在目录 {self.model_dir} 中未找到模型文件。")
        
        logging.info(f"正在从 {self.model_dir} 加载 {len(model_paths)} 个模型...")
        return [joblib.load(path) for path in model_paths]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        使用加载的模型对测试数据进行预测。
        """
        all_predictions = []
        num_models = len(self.models)
        for i, model in enumerate(self.models):
            logging.info(f"使用模型 {i+1}/{num_models} 进行预测...")
            y_pred_col = model.predict(X_test)
            all_predictions.append(y_pred_col)
        return np.column_stack(all_predictions)