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

    # 初始化一个空的 DataFrame
    combined_df = pd.DataFrame(index=df.index.unique())

    # 遍历每个证券代码，将其数据添加为新列
    for code in unique_codes:
        code_df = df[df['证券代码'] == code].copy()
        columns_to_keep = ['开盘价', '最高价', '最低价', '收盘价', '结算价', '成交量', '持仓量']
        code_df = code_df[columns_to_keep]
        code_df = code_df.add_prefix(f'{product_name}_')
        if code != product_code:
            # 如果不是主力合约，重命名列以区分
            code_df = code_df.add_prefix(f'{code}_')
        combined_df = combined_df.join(code_df, how='outer')

    # 生成新字段
    # 期限结构: 近远月价差、价差斜率
    if product_code in unique_codes:
        main_contract = df[df['证券代码'] == product_code]
        for code in unique_codes:
            if code != product_code:
                other_contract = df[df['证券代码'] == code]
                spread = main_contract['收盘价'] - other_contract['收盘价']
                combined_df[f'{product_name}_{code}_价差'] = spread
                combined_df[f'{product_name}_{code}_价差斜率'] = spread.diff()

    # 波动性: 日内波动率、价差波动率
    for code in unique_codes:
        code_df = df[df['证券代码'] == code]
        intraday_volatility = (code_df['最高价'] - code_df['最低价']) / code_df['收盘价']
        combined_df[f'{product_name}_{code}_日内波动率'] = intraday_volatility

    # 资金情绪: 持仓量变化
    for code in unique_codes:
        code_df = df[df['证券代码'] == code]
        open_interest_change = code_df['持仓量'].diff()
        combined_df[f'{product_name}_{code}_持仓量变化'] = open_interest_change

    # 相对动量: 合约相对强度
    for code in unique_codes:
        code_df = df[df['证券代码'] == code]
        relative_strength = code_df['收盘价'] / code_df['收盘价'].rolling(window=5).mean()
        combined_df[f'{product_name}_{code}_相对强度'] = relative_strength

    # 结算特性: 收盘结算偏离度
    for code in unique_codes:
        code_df = df[df['证券代码'] == code]
        settlement_deviation = (code_df['收盘价'] - code_df['结算价']) / code_df['结算价']
        combined_df[f'{product_name}_{code}_收盘结算偏离度'] = settlement_deviation

    # 换月特征: 新老主力价差
    if product_code in unique_codes:
        main_contract = df[df['证券代码'] == product_code]
        for code in unique_codes:
            if code != product_code:
                other_contract = df[df['证券代码'] == code]
                rollover_spread = main_contract['收盘价'] - other_contract['收盘价']
                combined_df[f'{product_name}_{code}_换月价差'] = rollover_spread

    # 市场宽度: 上涨合约占比
    daily_up_count = df.groupby(df.index)['收盘价'].apply(lambda x: (x.diff() > 0).sum())
    total_count = df.groupby(df.index)['收盘价'].count()
    market_breadth = daily_up_count / total_count
    combined_df[f'{product_name}_市场宽度'] = market_breadth

    return combined_df

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