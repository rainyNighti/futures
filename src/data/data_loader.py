import pandas as pd
import os
from typing import Dict, List

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """清理可能包含逗号数值列: 100,000,000"""
    if series.dtype == 'object':
        series = series.str.replace(',', '', regex=False)
    return pd.to_numeric(series, errors='coerce')

def preprocess_fundamental_data(file_path: str) -> pd.DataFrame:
    """加载并预处理单个基本面数据文件"""
    df = pd.read_csv(file_path)
    df = df.rename(columns={df.columns[0]: '日期'})
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期']) # 移除日期转换失败的行
    df = df.set_index('日期')
    for col in df.columns:
        df[col] = clean_numeric_column(df[col])
    df = df.dropna(axis=1, how='all')   # 有些csv文件末尾可能有空的列
    return df

def preprocess_trade_data(file_path: str, product_code: str, product_name: str) -> pd.DataFrame:
    """加载并预处理单个品种的交易数据，只保留主力合约"""
    df = pd.read_csv(file_path, parse_dates=['日期'])
    # 筛选主力合约
    df_main = df[df['证券代码'] == product_code].copy()
    df_main = df_main.set_index('日期')
    columns_to_keep = ['开盘价', '最高价', '最低价', '收盘价', '结算价', '成交量', '持仓量']
    df_main = df_main[columns_to_keep]
    df_main = df_main.add_prefix(f'{product_name}_')
    return df_main

def assemble_data(base_data_dir: str, trade_data_config: Dict, fundamental_paths: List[str]) -> pd.DataFrame:
    """
    根据配置加载、预处理并合并所有数据源。

    Args:
        base_data_dir (str): 数据文件的根目录.
        trade_data_config (Dict): 包含交易数据路径和代码的字典.
        fundamental_paths (List[str]): 基本面数据文件的相对路径列表.

    Returns:
        pd.DataFrame: 合并并初步处理后的DataFrame.
    """
    trade_dfs = []
    for name, config in trade_data_config.items():
        full_path = os.path.join(base_data_dir, config['path'])
        trade_dfs.append(preprocess_trade_data(full_path, config['code'], name))
    
    merged_trade_df = pd.concat(trade_dfs, axis=1)

    fundamental_dfs = []
    for rel_path in fundamental_paths:
        full_path = os.path.join(base_data_dir, rel_path)
        fundamental_dfs.append(preprocess_fundamental_data(full_path))

    merged_fundamental_df = pd.concat(fundamental_dfs, axis=1)
    
    combined_df = merged_trade_df.join(merged_fundamental_df, how='outer')
    combined_df = combined_df.sort_index()  # 按日期排序，确保前向填充的正确性

    ffill_cols = merged_fundamental_df.columns  # 仅对fundamental数据执行前向填充
    combined_df[ffill_cols] = combined_df[ffill_cols].ffill()
    
    # 只保留原始的交易日
    final_df = combined_df.loc[merged_trade_df.index].copy()
    
    return final_df