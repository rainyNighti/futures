from typing import Dict, Callable
import pandas as pd

def add_historical_features(df: pd.DataFrame, n_lags_d: int) -> pd.DataFrame:
    """
    为一个DataFrame的每一行添加前 N 帧（行）的数据作为新特征列。

    参数:
    df (pd.DataFrame): 输入的DataFrame，每一行代表一帧数据。
    n_lags_d (int): 要回顾的历史帧数。必须是正整数。

    返回:
    pd.DataFrame: 一个新的DataFrame，包含了原始数据以及历史数据作为新的列。
                  由于开头的行没有足够的历史数据，对应的新列将被填充为 NaN。
    """
    original_df = df.copy()
    
    # 循环创建 N 个滞后版本
    dfs_to_concat = [original_df]
    for i in range(1, n_lags_d + 1):
        shifted_df = original_df.shift(i)   # 将数据向下移动 i 行
        shifted_df.columns = [f'{col}_lag_{i}' for col in original_df.columns]
        dfs_to_concat.append(shifted_df)
    result_df = pd.concat(dfs_to_concat, axis=1)
    return result_df

def aggregate_major_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """将所有相关证券代码的数据并到列中，并生成新字段"""
    unique_codes = sorted(df['证券代码'].unique())
    main_code = unique_codes[0]
    # spot_code = unique_codes[1]
    monthly_codes = unique_codes[2:]

    columns_to_process = ['开盘价', '最高价', '最低价', '收盘价', '结算价', '成交量', '持仓量']

    # 主力合约
    df_main = df[df['证券代码'] == main_code][columns_to_process].add_suffix('_主力合约')

    # 只取月度合约
    df_monthly = df[df['证券代码'].isin(monthly_codes)].copy()
    
    # 按日期和成交量排序
    df_monthly_sorted = df_monthly.sort_values(by=[df_monthly.index.name or 'index', '成交量'], ascending=[True, False])
    grouped_by_date = df_monthly_sorted.groupby(df_monthly_sorted.index)
    
    # 获取次大合约
    df_second = grouped_by_date.nth(1)[columns_to_process].add_suffix('_次大合约')
    
    # 获取第三大合约
    df_third = grouped_by_date.nth(2)[columns_to_process].add_suffix('_第三大合约')
    combined_df = pd.concat([df_main, df_second, df_third], axis=1)

    return combined_df

def create_target_feature(df: pd.DataFrame, source_column: str, target_column: str) -> pd.DataFrame:
    """
        为表格添加：对应的y列
        target_column: ['T_5', 'T_10', 'T_20'] 三者中的一个
    """
    df_new = df.copy()
    df_new = df_new.sort_index()
    index = int(target_column.split('_')[-1])
    df_new[target_column] = df_new[source_column].shift(-index)
    return df_new

# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_TRADE: Dict[str, Callable] = {
    'aggregate_major_contracts': aggregate_major_contracts,
    'add_historical_features': add_historical_features,
    'create_target_feature': create_target_feature
}