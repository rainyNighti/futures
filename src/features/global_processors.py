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
    
# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_GLOBAL: Dict[str, Callable] = {
    # 1 魔法列处理
    "add_date_info": add_date_info,
    # 2 非数值列信息编码，目前不需要，因为所有列都是数值列，后续有了再更新
    # "label_encode": label_encode,

    # 3 缺失值处理
    "fill_with_stat": fill_with_stat,
    "fill_with_value": fill_with_value,
    "handle_missing_values_for_oil_data": handle_missing_values_for_oil_data,    # 定制化缺失值处理

    # 选择模型训练的列
    "update_used_fields": update_used_fields,
}