import numpy as np
from typing import Dict

class Evaluator:
    def __init__(self, eval_config: dict, dataset_config: dict):
        self.weights = eval_config.weights
        self.future_steps = dataset_config.future_steps
        self.num_target_vars = len(dataset_config.target_columns)

    def _calculate_pps(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算价格精准度 (Price Proximity Score, PPS)"""
        epsilon = 1e-9
        # BUG 这里官方的公式不开平方,需要邮件问一下官方
        rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + epsilon))))
        return 1 / (1 + rmspe)

    def calculate_final_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算最终得分，动态地根据配置解析y的结构。
        """
        results = {}
        final_score = 0.0
        
        for i, step in enumerate(self.future_steps):
            start_col = i * self.num_target_vars
            end_col = (i + 1) * self.num_target_vars
            
            pps = self._calculate_pps(y_true[:, start_col:end_col], y_pred[:, start_col:end_col])
            
            time_label = f't{step}'
            pps_label = f'PPS_T+{step}'
            
            results[pps_label] = pps
            final_score += self.weights[time_label] * pps
            
        results['Final_Score'] = final_score
        return results