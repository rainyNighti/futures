import pandas as pd
from typing import Dict, Callable

FREQUENCY_MAP_DEMAND = {
    '原油：加工量：中国（月）': 'M', 
    '原油：加工量：中国：独立炼厂（周）': 'W', 
    '原油：加工量：日本（周）': 'W', 
    '原油：加工量：美国（周）': 'W', 
    '原油：加工量：中国（周）': 'W', 
    '原油：加工量：中国：主营炼厂（周）': 'W', 
    '原油：表观消费量：中国（月）': 'M', 
    '原油：常减压：独立炼厂：产能利用率：中国（周）': 'W', 
    '原油：产能利用率：美国（周）': 'W', 
    '原油：产能利用率：日本（周）': 'W', 
    '原油：需求量：全球（月）': 'M', 
    '原油：主营炼厂：产能利用率：中国（周）': 'W'
}
def convert_column_name(name: str, lag=2) -> str:
    """
    lag_1 is the original data.
    lag_2 is the data found in 1 most recent (M, W, Y).
    """
    column_type = FREQUENCY_MAP_DEMAND[name]
    return f"{name}_{column_type}_lag_{lag}"

def add_supply_demand_balance(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    demand = df[convert_column_name('原油：需求量：全球（月）', 2)].fillna(0)
    supply_china = df[convert_column_name('原油：加工量：中国（周）', 2)].fillna(0)
    supply_american = df[convert_column_name('原油：加工量：美国（周）', 2)].fillna(0)
    supply_japan = df[convert_column_name('原油：加工量：日本（周）', 2)].fillna(0)
    df['加工量需求量比'] = demand / (supply_china  + supply_american + supply_japan + 1e-9)
    return df

def add_china_independent_percentage(df:pd.DataFrame, **kwargs) -> pd.DataFrame:
    supply_china_independent = df[convert_column_name('原油：加工量：中国：独立炼厂（周）', 2)].fillna(0)
    supply_china = df[convert_column_name('原油：加工量：中国（周）', 2)].fillna(0)
    supply_american = df[convert_column_name('原油：加工量：美国（周）', 2)].fillna(0)
    supply_japan = df[convert_column_name('原油：加工量：日本（周）', 2)].fillna(0)
    df['中国加工占比'] = supply_china / (supply_china  + supply_american + supply_japan + 1e-9)
    # 把supply_china_independent中supply_china为0的行全设为0
    supply_china_independent.loc[supply_china == 0] = 0
    df['中国独立炼厂占比'] = supply_china_independent / (supply_china + 1e-9)
    return df

def add_weighted_production_capability(df:pd.DataFrame, **kwargs) -> pd.DataFrame:
    production_capability_china_independent = df[convert_column_name('原油：常减压：独立炼厂：产能利用率：中国（周）', 2)].fillna(0)
    production_capability_american = df[convert_column_name('原油：产能利用率：美国（周）', 2)].fillna(0)
    production_capability_japan = df[convert_column_name('原油：产能利用率：日本（周）', 2)].fillna(0)
    production_capability_china_main = df[convert_column_name('原油：主营炼厂：产能利用率：中国（周）', 2)].fillna(0)
    df['平均产能利用率'] = (production_capability_china_independent + production_capability_american + production_capability_japan + production_capability_china_main) / 4
    return df


# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_DEMAND: Dict[str, Callable] = {
    "add_supply_demand_balance": add_supply_demand_balance,
    "add_china_independent_percentage": add_china_independent_percentage,
    "add_weighted_production_capability": add_weighted_production_capability
}