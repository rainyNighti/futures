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
    """加载并预处理单个品种的交易数据，将所有相关证券代码的数据并到列中，并生成新字段"""
    df = pd.read_csv(file_path, parse_dates=['日期'])
    df = df.set_index('日期')

    # 获取所有唯一的证券代码
    unique_codes = df['证券代码'].unique()
    unique_codes.sort()
    main_code = unique_codes[0]
    spot_code = unique_codes[1]
    monthly_codes = unique_codes[2:]

    # 初始化一个空的 DataFrame
    combined_df = pd.DataFrame(index=df.index.unique())

    # 计算每一天的次大和第三大月度合约（按成交量排序）
    # 只考虑monthly_codes
    columns_to_keep = ['开盘价', '最高价', '最低价', '收盘价', '结算价', '成交量', '持仓量']
    for date, group in df.groupby(df.index):
        # 主力合约
        main_group = group[group['证券代码'] == main_code]
        if not main_group.empty:
            main_row = main_group.iloc[0]
            for col in columns_to_keep:
                combined_df.at[date, f'{product_name}_主力合约_{col}'] = main_row[col]

        # 只取月度合约
        monthly_group = group[group['证券代码'].isin(monthly_codes)]
        if len(monthly_group) < 2:
            continue  # 没有足够的月度合约
        # 按成交量降序排序
        sorted_monthly = monthly_group.sort_values('成交量', ascending=False)
        # 取次大和第三大合约
        if len(sorted_monthly) >= 2:
            second_row = sorted_monthly.iloc[1]
            for col in columns_to_keep:
                combined_df.at[date, f'{product_name}_次大合约_{col}'] = second_row[col]
        if len(sorted_monthly) >= 3:
            third_row = sorted_monthly.iloc[2]
            for col in columns_to_keep:
                combined_df.at[date, f'{product_name}_第三大合约_{col}'] = third_row[col]

    return combined_df

def assemble_data(base_data_dir: str, trade_data_config: Dict, fundamental_paths: List[str], product_name: str) -> pd.DataFrame:
    """
    根据配置加载、预处理并合并所有数据源。

    Args:
        base_data_dir (str): 数据文件的根目录.
        trade_data_config (Dict): 包含交易数据路径和代码的字典.
        fundamental_paths (List[str]): 基本面数据文件的相对路径列表.

    Returns:
        pd.DataFrame: 合并并初步处理后的DataFrame.
    """
    config = trade_data_config[product_name]
    full_path = os.path.join(base_data_dir, config['path'])
    trade_dfs = preprocess_trade_data(full_path, config['code'], product_name)

    fundamental_dfs = []
    for rel_path in fundamental_paths:
        full_path = os.path.join(base_data_dir, rel_path)
        fundamental_dfs.append(preprocess_fundamental_data(full_path))

    merged_fundamental_df = pd.concat(fundamental_dfs, axis=1)
    
    combined_df = trade_dfs.join(merged_fundamental_df, how='outer')
    combined_df = combined_df.sort_index()  # 按日期排序，确保前向填充的正确性

    ffill_cols = merged_fundamental_df.columns  # 仅对fundamental数据执行前向填充
    combined_df[ffill_cols] = combined_df[ffill_cols].ffill()
    
    # 只保留原始的交易日
    final_df = combined_df.loc[trade_dfs.index].copy()
    
    return final_df