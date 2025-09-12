import numpy as np

def calculate_pps(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算价格精准度 (Price Proximity Score, PPS)"""
    epsilon = 1e-9
    # BUG 这里官方的公式不开平方,需要邮件问一下官方
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + epsilon))))
    return 1 / (1 + rmspe)