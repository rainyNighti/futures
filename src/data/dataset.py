from typing import Tuple, List
import pandas as pd
import numpy as np

def create_supervised_dataset(
    df: pd.DataFrame,
    history_window: int,
    future_steps: List[int],
    target_column: List[str]
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
            future_data = df[[target_column]].iloc[i + step - 1].values
            current_y.extend(future_data)
        y.append(current_y)
        
    return np.array(X), np.array(y)

def create_and_split_supervised_dataset(
    df: pd.DataFrame, 
    dataset_config: dict,
    product_name: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    封装了数据集创建和划分的完整流程，参数由配置驱动。
    """
    X, y = create_supervised_dataset(
        df,
        history_window=dataset_config.history_window,
        future_steps=dataset_config.future_steps,
        target_column=product_name + "_" + dataset_config.target_column
    )

    split_strategy = getattr(dataset_config, 'train_val_test_split', 'holdout')
    test_split_ratio = getattr(dataset_config, 'test_split_ratio', 0.2)

    if split_strategy == 'time_series_cv':
        # 时间序列交叉验证，返回最后一个分割（可根据需要扩展返回所有分割）
        chunk_size = getattr(dataset_config, 'time_series_cv_chunk_size', 244)
        n_samples = len(X)
        x_trains, y_trains, x_tests, y_tests = [], [], [], []
        start = 0
        while start <= n_samples:
            split_start = start
            split_end = min(start + chunk_size, n_samples)
            n_split = split_end - split_start
            train_end = split_start + int(n_split * (1 - test_split_ratio))
            test_end = split_end
            X_train, y_train = X[start:train_end], y[start:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]
            x_trains.append(X_train)
            y_trains.append(y_train)
            x_tests.append(X_test)
            y_tests.append(y_test)
            start += chunk_size
        # concatenate all splits
        X_train = np.concatenate(x_trains, axis=0)
        y_train = np.concatenate(y_trains, axis=0)
        X_test = np.concatenate(x_tests, axis=0)
        y_test = np.concatenate(y_tests, axis=0)
        return X_train, y_train, X_test, y_test

    elif split_strategy == 'holdout':
        split_index = int(len(X) * (1 - getattr(dataset_config, 'test_split_ratio', 0.2)))
        X_train, y_train = X[:split_index], y[:split_index]
        X_test, y_test = X[split_index:], y[split_index:]
        return X_train, y_train, X_test, y_test
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")



def create_predict_dataset(
    df: pd.DataFrame,
    history_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """创建预测数据集和日期列表"""
    X = []
    DATE = []
    end_index = len(df)
    
    for i in range(history_window, end_index):
        start_window = i - history_window
        end_window = i
        window_data = df.iloc[start_window:end_window]
        history_data = window_data.values
        date = window_data.index[-1]
        # history_data = df.iloc[start_window:end_window].values
        X.append(history_data.flatten())
        DATE.append(date)
        
        
    return np.array(X), DATE


def generate_predict_dataset(
    df: pd.DataFrame, 
    dataset_config: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, DATE = create_predict_dataset(
        df,
        history_window=dataset_config.history_window,
    )
    return X, DATE
