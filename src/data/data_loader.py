import pandas as pd
from typing import Dict, List
from pathlib import Path

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
    df = df.dropna(axis=0, how='all')   # 删除那些没有数据的行
    return df

def load_data(base_data_dir: str, trade_data_path: str, fundamental_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    根据配置加载所有数据

    Args:
        base_data_dir (str): 数据文件的根目录.
        trade_data_path (str): 包含交易数据路径.
        fundamental_paths (List[str]): 基本面数据文件的路径.

    Returns:
        所有的数据集csv
    """
    base_dir_path  = Path(base_data_dir)
    full_trade_path = base_dir_path  / trade_data_path

    dfs = {}
    trade_data = pd.read_csv(str(full_trade_path), parse_dates=['日期'])
    dfs['trade'] = trade_data.set_index('日期')

    for rel_path in fundamental_paths:
        filepath = base_dir_path  / rel_path        
        dfs[filepath.stem] = preprocess_fundamental_data(str(filepath))

    # 统一按照日期排序
    dfs = {k: v.sort_index() for k, v in dfs.items()}

    return dfs