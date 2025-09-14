from typing import Dict, Callable, List
from .fill_missing_value import fill_with_stat, fill_with_value, handle_missing_values_for_oil_data
import pandas as pd

def add_date_info(df: pd.DataFrame) -> pd.DataFrame:
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    return df

def update_used_fields(df: pd.DataFrame, use_fields: List[str]) -> pd.DataFrame:
    return df[use_fields]

def add_volume_change(df: pd.DataFrame) -> pd.DataFrame:
    main = f"成交量_主力合约"
    sub = f"成交量_次大合约"
    df["主力成交量变化"] = df[main].diff()
    df["次主力成交量变化"] = df[sub].diff()
    return df

def add_intraday_volatility(df: pd.DataFrame) -> pd.DataFrame:
    main_vol = (df["最高价_主力合约"] - df["最低价_主力合约"]) / (df["开盘价_主力合约"] + 1e-9)
    sub_vol = (df["最高价_次大合约"] - df["最低价_次大合约"]) / (df["开盘价_次大合约"] + 1e-9)
    df["主力日内波动率"] = main_vol
    df["次主力日内波动率"] = sub_vol
    return df

def add_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    main = f"收盘价_主力合约"
    sub = f"收盘价_次大合约"
    main_ret = df[main].pct_change()
    sub_ret = df[sub].pct_change()
    df["主力收益率"] = main_ret
    df["次主力收益率"] = sub_ret
    df["价格相对强弱"] = main_ret - sub_ret
    return df


####### 一批量化的指标



# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_GLOBAL: Dict[str, Callable] = {
    # 1 魔法列处理
    "add_date_info": add_date_info,
    'add_volume_change': add_volume_change,
    'add_intraday_volatility': add_intraday_volatility,
    'add_relative_strength': add_relative_strength,

    # 2 非数值列信息编码，目前不需要，因为所有列都是数值列，后续有了再更新
    # "label_encode": label_encode,

    # 3 缺失值处理
    "fill_with_stat": fill_with_stat,
    "fill_with_value": fill_with_value,
    "handle_missing_values_for_oil_data": handle_missing_values_for_oil_data,    # 定制化缺失值处理

    # 选择模型训练的列
    "update_used_fields": update_used_fields,
}