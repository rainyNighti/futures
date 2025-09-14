import pandas as pd
from typing import Dict, List, Callable
from copy import deepcopy

from .processors.trade_processor import PREPROCESSING_FUNCTIONS_TRADE
from .processors.supply_processor import FREQUENCY_MAP_SUPPLY, PREPROCESSING_FUNCTIONS_SUPPLY
from .processors.macro_processor import FREQUENCY_MAP_MACRO, PREPROCESSING_FUNCTIONS_MACRO
from .processors.fund_holding_processor import FREQUENCY_MAP_FUND_HOLDING, PREPROCESSING_FUNCTIONS_FUND_HOLDING
from .processors.inventory_processor import FREQUENCY_MAP_FUND_INVENTORY, PREPROCESSING_FUNCTIONS_FUND_INVENTORY
from .processors.profit_processor import FREQUENCY_MAP_PROFIT, PREPROCESSING_FUNCTIONS_PROFIT
from .processors.demand_processor import FREQUENCY_MAP_DEMAND, PREPROCESSING_FUNCTIONS_DEMAND
from .global_processors import PREPROCESSING_FUNCTIONS_GLOBAL

from src.utils.processor import merge_with_lags_multi_freq

_TABLE_PROCESSORS = {
    '基本面数据_供应': FREQUENCY_MAP_SUPPLY,
    '基本面数据_宏观': FREQUENCY_MAP_MACRO,
    '基本面数据_基金持仓': FREQUENCY_MAP_FUND_HOLDING,
    '基本面数据_库存': FREQUENCY_MAP_FUND_INVENTORY,
    '基本面数据_利润': FREQUENCY_MAP_PROFIT,
    '基本面数据_需求': FREQUENCY_MAP_DEMAND,
}

def _process_data(
    df: pd.DataFrame, 
    pipeline_config: List[Dict], 
    processing_functions: Dict[str, Callable]
) -> pd.DataFrame:
    """
    根据配置动态执行预处理流程。

    Args:
        df (pd.DataFrame): DataFrame.
        pipeline_config (List[Dict]): 来自配置文件的预处理步骤列表.
            约定：type表示函数名，其余为函数参数

    Returns:
        pd.DataFrame: 处理完成的DataFrame.
    """
    processed_df = df.copy()
    for params in pipeline_config:
        _type = params.pop('type', None)
        if _type:
            processed_df = processing_functions[_type](processed_df, **params)
    return processed_df

def feature_engineering_pipeline(
    df: pd.DataFrame, 
    pipeline_config: Dict, 
) -> pd.DataFrame:
    """
    根据配置动态执行特征工程流程。
    """
    copy_df = df.copy()
    config = deepcopy(pipeline_config)

    all_functions = {
        ** PREPROCESSING_FUNCTIONS_SUPPLY,
        ** PREPROCESSING_FUNCTIONS_MACRO,
        ** PREPROCESSING_FUNCTIONS_FUND_HOLDING,
        ** PREPROCESSING_FUNCTIONS_FUND_INVENTORY,
        ** PREPROCESSING_FUNCTIONS_PROFIT,
        ** PREPROCESSING_FUNCTIONS_DEMAND,
        ** PREPROCESSING_FUNCTIONS_GLOBAL,
    }

    # 3 全局数据处理
    processed_df = _process_data(copy_df, config, all_functions)
    return processed_df


def preprocess_data(
    raw_dfs: Dict[str, pd.DataFrame], 
    preprocess_config: Dict, 
):
    config = deepcopy(preprocess_config)

    # 1 处理 交易数据
    trade_df_raw = raw_dfs.pop('trade')
    processed_df = _process_data(trade_df_raw, config.trade, PREPROCESSING_FUNCTIONS_TRADE)

    # 2 合并 基本数据
    for df_name, df_data in raw_dfs.items():
        frequency_map = _TABLE_PROCESSORS[df_name]
        processed_df = merge_with_lags_multi_freq(processed_df.copy(), df_data.copy(), frequency_map)

    return processed_df