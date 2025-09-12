from typing import Dict, Callable
from .fill_missing_value import fill_with_stat, fill_with_value, drop_all_nan_rows_and_y_nan_rows
import pandas as pd

def add_date_info(df: pd.DataFrame) -> pd.DataFrame:
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    return df

# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_GLOBAL: Dict[str, Callable] = {
    # 1 魔法列处理
    "add_date_info": add_date_info,


    # 2 非数值列信息编码，目前不需要，因为所有列都是数值列，后续有了再更新
    # "label_encode": label_encode,


    # 3 缺失值处理
    # TODO 考虑按照缺失值的 unique，以及缺失值占列的数量进行缺失值填补
    "drop_all_nan_rows_and_y_nan_rows": drop_all_nan_rows_and_y_nan_rows,

    "fill_with_stat": fill_with_stat,
    "fill_with_value": fill_with_value,
}