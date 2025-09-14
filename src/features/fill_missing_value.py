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

# --- 1. 使用指定值填补 ---
def fill_with_value(df: pd.DataFrame, value: Union[int, float] = -999, include_columns: List[str] = None, exclude_columns: List[str] = None) -> pd.DataFrame:
    df_copy = df.copy()
    columns_to_process = _get_columns_to_process(df_copy, include_columns, exclude_columns)        
    df_copy[columns_to_process] = df_copy[columns_to_process].fillna(value)
    return df_copy

# --- 2. 使用统计值填补（均值、中位数、众数） ---
def fill_with_stat(df: pd.DataFrame, method: str = 'mean', include_columns: List[str] = None, exclude_columns: List[str] = ['T_5', 'T_10', 'T_20']) -> pd.DataFrame:
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

# -- 3. 更新，定制化的填补方法 ---
def handle_missing_values_for_oil_data(df: pd.DataFrame, nan_threshold: float = 0.7, exclude_columns: List[str] = ['T_5', 'T_10', 'T_20'], interpolate_method: str = 'linear') -> pd.DataFrame:
    """
    针对原油高维时间序列数据，进行系统性的缺失值处理。

    处理流程:
    1. 深拷贝原始数据，避免修改原DataFrame。
    2. (全局删除) 删除缺失值比例超过阈值(nan_threshold)的列。
    3. (分类填充) 根据列名特征，将列分为'快变量'和'慢变量'。
       - 慢变量 (W, M, Q, Y): 使用向前填充 (ffill)。
       - 快变量 (D, 行情数据等): 使用线性插值 (interpolate)。
    4. (最终清理) 对整个DataFrame进行一次向后填充(bfill)，以处理数据集最开始可能存在的NaN。

    Args:
        df (pd.DataFrame): 包含缺失值的原始数据，索引应为时间序列。
        nan_threshold (float): 删除列的缺失率阈值，默认为0.7。

    Returns:
        pd.DataFrame: 经过完整缺失值处理后的数据。
    """
   
    # 1. 深拷贝，避免修改原始数据
    processed_df = df.copy()
    processed_df = processed_df.sort_index()

    # 2. 全局删除缺失过多的列
    initial_cols_count = len(processed_df.columns)
    missing_ratio = processed_df.isnull().sum() / len(processed_df)
    cols_to_drop = missing_ratio[missing_ratio > nan_threshold].index
    
    if len(cols_to_drop) > 0:
        processed_df.drop(columns=cols_to_drop, inplace=True)
        print(f"步骤1: 删除了 {len(cols_to_drop)} 个缺失率超过 {nan_threshold:.0%} 的列。")
        for col in cols_to_drop:
            print(f"  - {col}")
    
    # 3. 根据列名进行分类
    slow_cols = [] # 使用 ffill
    fast_cols = [] # 使用 interpolate

    # 设定更新很快的数据列名，这些数据更新很快，用插值法
    market_data_keywords = ['开盘价', '最高价', '最低价', '收盘价', '结算价', '成交量', '持仓量']
    
    for col in processed_df.columns:
        # 规则1: W, M, Q, Y 后缀的为慢变量
        if '_W_' in col or '_M_' in col or '_Q_' in col or '_Y_' in col:
            slow_cols.append(col)
        # 规则2: D 后缀或包含行情关键词的为快变量
        elif '_D_' in col or any(keyword in col for keyword in market_data_keywords):
            fast_cols.append(col)
        # 规则3: 其他未分类的，作为快变量处理更为稳妥
        else:
            print(f"对这些变量不进行填充：{col}")

    # 4. 执行分类填充
    # 对慢变量使用ffill
    if slow_cols:
        processed_df[slow_cols] = processed_df[slow_cols].ffill()

    # 对快变量使用interpolate
    if fast_cols:
        # 使用线性插值，并且只从前向后插值
        processed_df[fast_cols] = processed_df[fast_cols].interpolate(method=interpolate_method, limit_direction='forward')
        # print("步骤3b: 对快变量执行 线性插值(interpolate) 完成。")

    # 5. 最终清理：处理数据开头可能存在的NaN
    # 经过插值和ffill后，只有数据最开始的部分可能还有NaN
    # 使用bfill可以有效地从后往前把“已知”信息填充到开头
    total_columns = fast_cols + slow_cols
    initial_nan_count = processed_df.isnull().sum().sum()
    if initial_nan_count > 0:
        processed_df[total_columns] = processed_df[total_columns].bfill()
        # final_nan_count = processed_df.isnull().sum().sum()
        # print(f"\n步骤4: 执行全局向后填充(bfill)清理头部NaN，清除了 {initial_nan_count - final_nan_count} 个NaN。")

    # 检查是否还有任何NaN
    remaining_nans = processed_df.isnull().sum().sum()
    if remaining_nans != 0:
        print(f"\n--- 警告：处理后仍有 {remaining_nans} 个缺失值，请检查数据！ ---")
        
    return processed_df