import pandas as pd
from typing import List, Dict, Callable

def fill_miss_value(df: pd.DataFrame) -> pd.DataFrame:
    """使用前向和后向填充处理缺失值。"""
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def frequency_encode_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
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
}

def execute_preprocessing_pipeline(df: pd.DataFrame, pipeline_config: List[Dict]) -> pd.DataFrame:
    """
    根据配置动态执行预处理流程。

    Args:
        df (pd.DataFrame): 待处理的DataFrame.
        pipeline_config (List[Dict]): 来自配置文件的预处理步骤列表.

    Returns:
        pd.DataFrame: 处理完成的DataFrame.
    """
    processed_df = df.copy()
    for step in pipeline_config:
        step_name = step['name']
        if step_name in _PREPROCESSING_FUNCTIONS:
            # params = step.get('params', {}) # 如果函数需要参数
            processed_df = _PREPROCESSING_FUNCTIONS[step_name](processed_df)
        else:
            raise ValueError(f"未知的预处理步骤: {step_name}")
    return processed_df