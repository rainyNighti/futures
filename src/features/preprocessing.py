import pandas as pd
from typing import List, Dict, Callable

def fill_miss_value(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """使用前向和后向填充处理缺失值。"""
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def frequency_encode_non_numeric(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """对DataFrame中的非数值列进行频率编码。"""
    df_encoded = df.copy()
    non_numeric_cols = df_encoded.select_dtypes(exclude=['number', 'bool']).columns
    for col in non_numeric_cols:
        freq_map = df_encoded[col].value_counts(normalize=True)
        df_encoded[col] = df_encoded[col].map(freq_map)
    return df_encoded


# 注册所有可用的预处理函数
_PREPROCESSING_FUNCTIONS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    'fill_miss_value': fill_miss_value,
    'frequency_encode_non_numeric': frequency_encode_non_numeric,
    # 'scale_numeric_features': scale_numeric_features, # 新函数在这里注册
    'add_close_spread': lambda df, **kwargs: add_close_spread(df, kwargs['product_name']),
    'add_settle_spread': lambda df, **kwargs: add_settle_spread(df, kwargs['product_name']),
    'add_highlow_spread_range': lambda df, **kwargs: add_highlow_spread_range(df, kwargs['product_name']),
    'add_open_gap_spread': lambda df, **kwargs: add_open_gap_spread(df, kwargs['product_name']),
    'add_volume_ratio': lambda df, **kwargs: add_volume_ratio(df, kwargs['product_name']),
    'add_volume_change': lambda df, **kwargs: add_volume_change(df, kwargs['product_name']),
    'add_volume_direction': lambda df, **kwargs: add_volume_direction(df, kwargs['product_name']),
    'add_oi_change': lambda df, **kwargs: add_oi_change(df, kwargs['product_name']),
    'add_oi_transfer': lambda df, **kwargs: add_oi_transfer(df, kwargs['product_name']),
    'add_oi_ratio': lambda df, **kwargs: add_oi_ratio(df, kwargs['product_name']),
    'add_intraday_volatility': lambda df, **kwargs: add_intraday_volatility(df, kwargs['product_name']),
    'add_relative_strength': lambda df, **kwargs: add_relative_strength(df, kwargs['product_name']),
}

# 1. 价格类属性
def add_close_spread(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main = f"{product_name}_主力合约_收盘价"
    sub = f"{product_name}_次大合约_收盘价"
    df[f"{product_name}_收盘价差"] = df[main] - df[sub]
    return df

def add_settle_spread(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main = f"{product_name}_主力合约_结算价"
    sub = f"{product_name}_次大合约_结算价"
    df[f"{product_name}_结算价差"] = df[main] - df[sub]
    return df

def add_highlow_spread_range(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main_high = f"{product_name}_主力合约_最高价"
    sub_high = f"{product_name}_次大合约_最高价"
    main_low = f"{product_name}_主力合约_最低价"
    sub_low = f"{product_name}_次大合约_最低价"
    high_spread = df[main_high] - df[sub_high]
    low_spread = df[main_low] - df[sub_low]
    df[f"{product_name}_价差区间"] = high_spread - low_spread
    return df

def add_open_gap_spread(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main_open = f"{product_name}_主力合约_开盘价"
    sub_open = f"{product_name}_次大合约_开盘价"
    main_close = f"{product_name}_主力合约_收盘价"
    sub_close = f"{product_name}_次大合约_收盘价"
    today_gap = df[main_open] - df[sub_open]
    yesterday_gap = df[main_close].shift(1) - df[sub_close].shift(1)
    df[f"{product_name}_开盘跳空价差"] = today_gap - yesterday_gap
    return df

# 2. 成交量类属性
def add_volume_ratio(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main = f"{product_name}_主力合约_成交量"
    sub = f"{product_name}_次大合约_成交量"
    df[f"{product_name}_成交量比率"] = df[main] / (df[sub] + 1e-9)
    return df

def add_volume_change(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main = f"{product_name}_主力合约_成交量"
    sub = f"{product_name}_次大合约_成交量"
    df[f"{product_name}_主力成交量变化"] = df[main].diff()
    df[f"{product_name}_次主力成交量变化"] = df[sub].diff()
    return df

def add_volume_direction(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    # 价差扩大/缩小时，哪个合约成交量更大
    close_spread = f"{product_name}_收盘价差"
    main_vol = f"{product_name}_主力合约_成交量"
    sub_vol = f"{product_name}_次大合约_成交量"
    # 先确保有收盘价差
    if close_spread not in df.columns:
        df = add_close_spread(df, product_name)
    spread_change = df[close_spread].diff()
    df[f"{product_name}_放量方向"] = df.apply(
        lambda row: '主力' if (row[main_vol] > row[sub_vol] and row[close_spread] > 0) else ('次主力' if (row[main_vol] < row[sub_vol] and row[close_spread] < 0) else '无'), axis=1)
    return df

# 3. 持仓量类属性
def add_oi_change(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main = f"{product_name}_主力合约_持仓量"
    df[f"{product_name}_主力持仓量变化"] = df[main].diff()
    df[f"{product_name}_主力持仓量变化率"] = df[main].pct_change()
    return df

def add_oi_transfer(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main = f"{product_name}_主力合约_持仓量"
    sub = f"{product_name}_次大合约_持仓量"
    main_dec = -df[main].diff()
    sub_inc = df[sub].diff()
    df[f"{product_name}_持仓量转移"] = sub_inc - main_dec
    return df

def add_oi_ratio(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main = f"{product_name}_主力合约_持仓量"
    sub = f"{product_name}_次大合约_持仓量"
    df[f"{product_name}_持仓量比率"] = df[main] / (df[sub] + 1e-9)
    return df

# 4. 衍生指标
def add_intraday_volatility(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main_vol = (df[f"{product_name}_主力合约_最高价"] - df[f"{product_name}_主力合约_最低价"]) / (df[f"{product_name}_主力合约_开盘价"] + 1e-9)
    sub_vol = (df[f"{product_name}_次大合约_最高价"] - df[f"{product_name}_次大合约_最低价"]) / (df[f"{product_name}_次大合约_开盘价"] + 1e-9)
    df[f"{product_name}_主力日内波动率"] = main_vol
    df[f"{product_name}_次主力日内波动率"] = sub_vol
    return df

def add_relative_strength(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    main = f"{product_name}_主力合约_收盘价"
    sub = f"{product_name}_次大合约_收盘价"
    main_ret = df[main].pct_change()
    sub_ret = df[sub].pct_change()
    df[f"{product_name}_主力收益率"] = main_ret
    df[f"{product_name}_次主力收益率"] = sub_ret
    df[f"{product_name}_价格相对强弱"] = main_ret - sub_ret
    return df

def execute_preprocessing_pipeline(df: pd.DataFrame, pipeline_config: List[Dict], product_name: str) -> pd.DataFrame:
    """
    根据配置动态执行预处理流程。

    Args:
        df (pd.DataFrame): 待处理的DataFrame.
        pipeline_config (List[Dict]): 来自配置文件的预处理步骤列表.

    Returns:
        pd.DataFrame: 处理完成的DataFrame.
    """
    processed_df = df.copy()
    kwargs = {'product_name': product_name}
    for step in pipeline_config:
        step_name = step['name']
        if step_name in _PREPROCESSING_FUNCTIONS:
            # params = step.get('params', {}) # 如果函数需要参数
            processed_df = _PREPROCESSING_FUNCTIONS[step_name](processed_df, **kwargs)
        else:
            raise ValueError(f"未知的预处理步骤: {step_name}")
    return processed_df