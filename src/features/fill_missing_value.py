import pandas as pd
import numpy as np
from typing import List, Union

def _get_columns_to_process(df: pd.DataFrame, include_columns: list = None, exclude_columns: list = None) -> list:
    """内部辅助函数，用于根据包含和排除规则确定要处理的列列表。"""
    if include_columns:
        columns_to_process = list(include_columns)
    else:
        # 自动检测所有数值类型的列
        columns_to_process = df.select_dtypes(include=np.number).columns.tolist()
    if exclude_columns:
        # 从待处理列表中排除指定的列
        columns_to_process = [col for col in columns_to_process if col not in exclude_columns]
    return columns_to_process

def drop_all_nan_rows_and_y_nan_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """ 
        删除所有值均为NaN的行 和 y值是NaN的行。
        如果一行中所有值都是 NaN，那么y也必然是 NaN。所以只需要删y为 NaN 的行。
    """
    return df.dropna(subset=[target_column])

# --- 1. 使用指定值填补 ---
def fill_with_value(df: pd.DataFrame, value: Union[int, float] = -999, include_columns: List[str] = None, exclude_columns: List[str] = None) -> pd.DataFrame:
    df_copy = df.copy()
    columns_to_process = _get_columns_to_process(df_copy, include_columns, exclude_columns)        
    df_copy[columns_to_process] = df_copy[columns_to_process].fillna(value)
    return df_copy

# --- 2. 使用统计值填补（均值、中位数、众数） ---
def fill_with_stat(df: pd.DataFrame, method: str = 'mean', include_columns: List[str] = None, exclude_columns: List[str] = None) -> pd.DataFrame:
    """
    使用统计值（'mean', 'median', 'mode'）填充缺失值。
    """
    df_copy = df.copy()
    columns_to_process = _get_columns_to_process(df_copy, include_columns, exclude_columns)
    
    if method == 'mean':
        fill_values = df_copy[columns_to_process].mean()
    elif method == 'median':
        fill_values = df_copy[columns_to_process].median()
    elif method == 'mode':
        # mode()可能返回多个值，我们取第一个
        fill_values = df_copy[columns_to_process].mode().iloc[0]
    else:
        raise ValueError("Method must be one of 'mean', 'median', or 'mode'")
        
    df_copy[columns_to_process] = df_copy[columns_to_process].fillna(fill_values)
    return df_copy