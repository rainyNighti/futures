from typing import Tuple, List
import pandas as pd
import numpy as np

def split_dataset(
    df: pd.DataFrame, 
    test_split_ratio: float,
    target_column: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = df[target_column]
    X = df.drop(columns=target_column)
    split_index = int(len(X) * (1 - test_split_ratio))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # .values 会返回 NumPy 数组表示
    return X_train.values, X_test.values, y_train.values, y_test.values

# TODO 更新predict函数
# def create_predict_dataset(
#     df: pd.DataFrame,
#     history_window: int,
#     future_steps: List[int],
#     target_columns: List[str]
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """创建预测数据集和日期列表"""
#     X = []
#     DATE = []
#     end_index = len(df)
    
#     for i in range(history_window, end_index):
#         start_window = i - history_window
#         end_window = i
#         window_data = df.iloc[start_window:end_window]
#         history_data = window_data.values
#         date = window_data.index[-1]
#         # history_data = df.iloc[start_window:end_window].values
#         X.append(history_data.flatten())
#         DATE.append(date)
        
        
#     return np.array(X), DATE


# def generate_predict_dataset(
#     df: pd.DataFrame, 
#     dataset_config: dict
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     X, DATE = create_predict_dataset(
#         df,
#         history_window=dataset_config.history_window,
#         future_steps=dataset_config.future_steps,
#         target_columns=dataset_config.target_columns
#     )
#     return X, DATE
