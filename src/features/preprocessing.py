import pandas as pd
from typing import Dict, List, Callable

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
    '基本面数据_供应': (FREQUENCY_MAP_SUPPLY, PREPROCESSING_FUNCTIONS_SUPPLY, 'supply_pipeline'),
    '基本面数据_宏观': (FREQUENCY_MAP_MACRO, PREPROCESSING_FUNCTIONS_MACRO, 'macro_pipeline'),
    '基本面数据_基金持仓': (FREQUENCY_MAP_FUND_HOLDING, PREPROCESSING_FUNCTIONS_FUND_HOLDING, 'fund_holding_pipeline'),
    '基本面数据_库存': (FREQUENCY_MAP_FUND_INVENTORY, PREPROCESSING_FUNCTIONS_FUND_INVENTORY, 'inventory_pipeline'),
    '基本面数据_利润': (FREQUENCY_MAP_PROFIT, PREPROCESSING_FUNCTIONS_PROFIT, 'profit_pipeline'),
    '基本面数据_需求': (FREQUENCY_MAP_DEMAND, PREPROCESSING_FUNCTIONS_DEMAND, 'demand_pipeline'),
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
        type = params.pop('type')
        processed_df = processing_functions[type](processed_df, **params)
    return processed_df

def _process_fundamental_data(
    base_df: pd.DataFrame, 
    df_to_merge: pd.DataFrame,
    frequency_map: Dict[str, str],
    pipeline_config: List[Dict], 
    processing_functions: Dict[str, Callable],
) -> pd.DataFrame:
    """
        base_df: 总表数据
        df_to_merge: 待合并的新表数据
        frequency_map: 记录fundamental的列数据的更新频率
    """
    merge_config = pipeline_config.pop(0)   # 第0个约定为合并函数
    processed_df = merge_with_lags_multi_freq(base_df.copy(), df_to_merge.copy(), frequency_map, **merge_config)
    return _process_data(processed_df, pipeline_config, processing_functions)


def execute_preprocessing_pipeline(
    raw_dfs: Dict[str, pd.DataFrame], 
    pipeline_config: Dict, 
    target_column: str
) -> pd.DataFrame:
    """
    根据配置动态执行预处理流程。
    """
    config = pipeline_config.copy()

    # 1 处理 交易数据
    trade_df_raw = raw_dfs.pop('trade')
    processed_df = _process_data(trade_df_raw, config.trade_pipeline, PREPROCESSING_FUNCTIONS_TRADE)

    # 2 逐个处理 & 合并 基本数据
    for df_name, df_data in raw_dfs.items():
        frequency_map, processing_functions, functions_name = _TABLE_PROCESSORS[df_name]
        processed_df = _process_fundamental_data(
            base_df=processed_df, 
            df_to_merge=df_data,
            frequency_map=frequency_map,
            pipeline_config=pipeline_config[functions_name],
            processing_functions=processing_functions,
        )

    # 3 全局数据处理
    processed_df = _process_data(processed_df, config.global_pipeline, PREPROCESSING_FUNCTIONS_GLOBAL)
    return processed_df