
from typing import Dict, List
import pandas as pd
import numpy as np
# BUG 推理时，如果使用了数值填充，需要将数值记录下来，推理时没有这么多行，会导致选取的值不准确
# TODO check所有表格的列的数据类型，使用to_numeric转换，过滤脏数据
# TODO check日期列转换是否正确，有很多错误的警告
def set_negative_to_nan(df: pd.DataFrame, exclude_columns: List[str] = []) -> pd.DataFrame:
    """
    将DataFrame中所有小于0的值替换为NaN，但会排除指定的列。
    Args:
        df (pd.DataFrame): 输入的pandas DataFrame。
        exclude_columns (List[str]): 一个字符串列表，包含不应处理的列名。
    """
    df_copy = df.copy()
    columns_to_process = [col for col in df_copy.columns if col not in exclude_columns]
    # 当条件（>= 0）为True时，它会保留原始值。当条件为False时，它会用NaN（默认）替换该值。
    df_copy[columns_to_process] = df_copy[columns_to_process].where(df_copy[columns_to_process] >= 0)
    return df_copy

def _clean_trade(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 对于sc需要先将'开盘价'这一列转为float64，有很多脏数据
    df['开盘价'] = pd.to_numeric(df['开盘价'], errors='coerce')

    # 2. 该表中的 0 都是NaN
    df = df.replace(0, np.nan)

    # 3. 清理不应该是负数的异常值
    df = set_negative_to_nan(df, ['ID', '证券代码'])
    return df

def _clean_supply(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 该表中无 0，未来新数据有0就是nan
    df = df.replace(0, np.nan)

    # 2. 清理不应该是负数的异常值
    df = set_negative_to_nan(df)
    return df

def _clean_macro(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 该表中部分列可以确定 0 都是 NaN，部分列不确定，保留
    columns_to_process = [
        '美国:拖欠率:信用卡消费贷款:所有商业银行:季调',
        '美国:拖欠率:房地产贷款:所有商业银行:季调',
        '中国:零售渗透率:新能源乘用车',
        '中国:工业企业:利润总额:当月同比',
        '中国:房地产开发投资完成额:累计同比',
        '中国:制造业PMI',
        '中国:社会融资规模:当月值'
    ]
    df[columns_to_process] = df[columns_to_process].replace(0, np.nan)

    # 2. 清理不应该是负数的异常值
    df = set_negative_to_nan(df, [
        '美国:CPI:同比', '美国:PPI:最终需求:同比:季调', '欧元区:HICP(调和CPI):当月同比', 
        '欧元区20国:PPI:当月同比', '中国:工业企业:利润总额:当月同比', '中国:CPI:当月同比', 
        '中国:PPI:全部工业品:当月同比', '中国:房地产开发投资完成额:累计同比',
        '中国:M1:同比', '中国:M2:同比'
    ])
    return df

def _clean_fund_holding(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 该表中的 0 都是NaN
    df = df.replace(0, np.nan)

    # 2. 清理不应该是负数的异常值
    df = set_negative_to_nan(df)
    return df

def _clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 该表中无0，后续有 0 都是NaN
    df = df.replace(0, np.nan)

    # 2. 清理不应该是负数的异常值
    df = set_negative_to_nan(df)
    return df

def _clean_profit(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 该表中数据都有可能为 0
    # 2. 该表中，数据都有可能是负数
    return df

def _clean_demand(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 该表中无0，后续有 0 都是NaN
    df = df.replace(0, np.nan)

    # 2. 清理不应该是负数的异常值
    df = set_negative_to_nan(df)
    return df

MAP_CLEAN_FUNCTION = {
    'trade': _clean_trade,
    '基本面数据_供应': _clean_supply,
    '基本面数据_宏观': _clean_macro,
    '基本面数据_基金持仓': _clean_fund_holding,
    '基本面数据_库存': _clean_inventory,
    '基本面数据_利润': _clean_profit,
    '基本面数据_需求': _clean_demand,
}

def clean_data(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    定制化对每张数据表进行清洗
    1. 处理各个表中，原本应该是nan的值，但用数字0替代
    TODO
    2. 清理异常值（离群值）
    3. 根据EDA，设置数据的范围，删除重复行，补正部分数据
    4. 执行数据清洗cleanlab
    """
    new_dfs = {}
    for k, v in dfs.items():
        new_dfs[k] = MAP_CLEAN_FUNCTION[k](v)
    return new_dfs
