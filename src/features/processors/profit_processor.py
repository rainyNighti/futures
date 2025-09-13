from typing import Dict, Callable


import pandas as pd

FREQUENCY_MAP_PROFIT = {
    '原油：常减压：独立炼厂：装置毛利：中国（周）': 'W',
    '原油：炼油工序：毛利：中国：主营炼厂（周）': 'W',
    '原油：炼油工序：毛利：山东：独立炼厂（周）': 'W',
    '柴油：裂解：价差：中国：主营销售公司（日）': 'D',
    '汽油：裂解：价差：中国：主营销售公司（日）': 'D'
}

def convert_column_name_profit(name: str, lag=2) -> str:
    column_type = FREQUENCY_MAP_PROFIT[name]
    return f"{name}_{column_type}_lag_{lag}"

def aggregate_daily_to_weekly(series: pd.Series) -> pd.Series:
    # 假设index为日期类型，聚合为周均值，结果对齐到周末
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    return series.resample('W').mean()

# 1. 炼油利润全景监控
def add_refining_profit_panorama(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # 取山东独立炼厂毛利（周）
    profit_col = convert_column_name_profit('原油：炼油工序：毛利：山东：独立炼厂（周）', 2)
    # 汽油、柴油裂解价差（日），需聚合为周
    gasoline_col = convert_column_name_profit('汽油：裂解：价差：中国：主营销售公司（日）', 2)
    diesel_col = convert_column_name_profit('柴油：裂解：价差：中国：主营销售公司（日）', 2)
    # 聚合日度为周度
    if gasoline_col in df.columns and diesel_col in df.columns:
        # 假设df有'date'列
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        # 聚合
        gasoline_weekly = aggregate_daily_to_weekly(df[gasoline_col].fillna(0))
        diesel_weekly = aggregate_daily_to_weekly(df[diesel_col].fillna(0))
        # 结果对齐
        df['汽油裂解价差_周均'] = gasoline_weekly
        df['柴油裂解价差_周均'] = diesel_weekly
        # 恢复index
        df = df.reset_index()
    # 组合字段
    df['炼油利润全景监控'] = df.get(profit_col, 0) + df.get('汽油裂解价差_周均', 0) + df.get('柴油裂解价差_周均', 0)
    return df

# 2. 炼厂竞争格局分析
def add_refinery_competition(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    profit_independent = convert_column_name_profit('原油：炼油工序：毛利：山东：独立炼厂（周）', 2)
    profit_main = convert_column_name_profit('原油：炼油工序：毛利：中国：主营炼厂（周）', 2)
    df['独立炼厂_vs_主营炼厂_毛利差'] = df.get(profit_independent, 0) - df.get(profit_main, 0)
    return df

# 3. 成品油裂解强弱对比
def add_gasoline_vs_diesel_spread(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    gasoline_col = convert_column_name_profit('汽油：裂解：价差：中国：主营销售公司（日）', 2)
    diesel_col = convert_column_name_profit('柴油：裂解：价差：中国：主营销售公司（日）', 2)
    # 聚合日度为周度
    if gasoline_col in df.columns and diesel_col in df.columns:
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        gasoline_weekly = aggregate_daily_to_weekly(df[gasoline_col].fillna(0))
        diesel_weekly = aggregate_daily_to_weekly(df[diesel_col].fillna(0))
        df['汽油裂解价差_周均'] = gasoline_weekly
        df['柴油裂解价差_周均'] = diesel_weekly
        df = df.reset_index()
    df['汽柴油裂解价差差'] = df.get('汽油裂解价差_周均', 0) - df.get('柴油裂解价差_周均', 0)
    return df

# 4. 加工毛利（独立炼厂）与成品油裂解价差（主营销售）的组合
def add_independent_profit_and_main_spread(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    profit_independent = convert_column_name_profit('原油：常减压：独立炼厂：装置毛利：中国（周）', 2)
    gasoline_col = convert_column_name_profit('汽油：裂解：价差：中国：主营销售公司（日）', 2)
    diesel_col = convert_column_name_profit('柴油：裂解：价差：中国：主营销售公司（日）', 2)
    # 聚合日度为周度
    if gasoline_col in df.columns and diesel_col in df.columns:
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        gasoline_weekly = aggregate_daily_to_weekly(df[gasoline_col].fillna(0))
        diesel_weekly = aggregate_daily_to_weekly(df[diesel_col].fillna(0))
        df['汽油裂解价差_周均'] = gasoline_weekly
        df['柴油裂解价差_周均'] = diesel_weekly
        df = df.reset_index()
    df['独立炼厂毛利_主营销售裂解价差组合'] = df.get(profit_independent, 0) + df.get('汽油裂解价差_周均', 0) + df.get('柴油裂解价差_周均', 0)
    return df

# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_PROFIT: Dict[str, Callable] = {
    "add_refining_profit_panorama": add_refining_profit_panorama,
    "add_refinery_competition": add_refinery_competition,
    "add_gasoline_vs_diesel_spread": add_gasoline_vs_diesel_spread,
    "add_independent_profit_and_main_spread": add_independent_profit_and_main_spread
}