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
