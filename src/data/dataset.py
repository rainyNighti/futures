from typing import Tuple, List
import pandas as pd
import numpy as np

def create_supervised_dataset(
    df: pd.DataFrame,
    history_window: int,
    future_steps: List[int],
    target_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """将时间序列DataFrame转换为监督学习的(X, y)格式。"""
    X, y = [], []
    end_index = len(df) - max(future_steps)
    
    for i in range(history_window, end_index):
        start_window = i - history_window
        end_window = i
        history_data = df.iloc[start_window:end_window].values
        X.append(history_data.flatten())
        
        current_y = []
        for step in future_steps:
            future_data = df[target_columns].iloc[i + step - 1].values
            current_y.extend(future_data)
        y.append(current_y)
        
    return np.array(X), np.array(y)

def create_and_split_supervised_dataset(
    df: pd.DataFrame, 
    dataset_config: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    封装了数据集创建和划分的完整流程，参数由配置驱动。
    """
    X, y = create_supervised_dataset(
        df,
        history_window=dataset_config.history_window,
        future_steps=dataset_config.future_steps,
        target_columns=dataset_config.target_columns
    )
    
    split_index = int(len(X) * (1 - dataset_config.test_split_ratio))
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    return X_train, y_train, X_test, y_test