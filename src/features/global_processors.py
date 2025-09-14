from typing import Dict, Callable, List
from .fill_missing_value import fill_with_stat, fill_with_value, handle_missing_values_for_oil_data
import pandas as pd
from src.utils.processor import get_lag_cols

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
    return df.copy()

def add_intraday_volatility(df: pd.DataFrame) -> pd.DataFrame:
    main_vol = (df["最高价_主力合约"] - df["最低价_主力合约"]) / (df["开盘价_主力合约"] + 1e-9)
    sub_vol = (df["最高价_次大合约"] - df["最低价_次大合约"]) / (df["开盘价_次大合约"] + 1e-9)
    df["主力日内波动率"] = main_vol
    df["次主力日内波动率"] = sub_vol
    return df.copy()

####### 一批量化的指标
def generate_rolling_stats_features(df: pd.DataFrame, max_lag: int = 60) -> pd.DataFrame:
    """
    为主要指标（收盘价, 成交量, 持仓量）生成滚动统计特征。
    """
    # 定义要计算的指标和窗口期
    metrics = ['收盘价_主力合约', '成交量_主力合约', '持仓量_主力合约']
    windows = [5, 10, 20, 40, 60]

    for metric in metrics:
        # 获取当前指标的所有历史lag列
        metric_cols = get_lag_cols(metric, max_lag)
        
        for w in windows:
            # 确保窗口期不超过总lag数
            if w <= max_lag:
                window_cols = metric_cols[:w]
                
                # 为了简化名称，我们使用英文缩写
                metric_name_short = ""
                if "收盘价" in metric:
                    metric_name_short = "close"
                elif "成交量" in metric:
                    metric_name_short = "volume"
                elif "持仓量" in metric:
                    metric_name_short = "oi" # Open Interest

                # 计算均值 (趋势)
                df[f'{metric_name_short}_rolling_mean_{w}'] = df[window_cols].mean(axis=1)
                
                # 计算标准差 (波动率)
                df[f'{metric_name_short}_rolling_std_{w}'] = df[window_cols].std(axis=1)
                
                # 计算最大值
                df[f'{metric_name_short}_rolling_max_{w}'] = df[window_cols].max(axis=1)
                
                # 计算最小值
                df[f'{metric_name_short}_rolling_min_{w}'] = df[window_cols].min(axis=1)
                
                # 计算偏度 (数据分布的对称性)
                df[f'{metric_name_short}_rolling_skew_{w}'] = df[window_cols].skew(axis=1)

                # 计算峰度 (数据分布的陡峭程度)
                df[f'{metric_name_short}_rolling_kurt_{w}'] = df[window_cols].kurt(axis=1)

    return df.copy()

def generate_momentum_features(df: pd.DataFrame, max_lag: int = 60) -> pd.DataFrame:
    """
    生成动量 (Momentum) 和变化率 (ROC) 特征。
    """
    # 我们主要关心收盘价
    base_metric = '收盘价_主力合约'
    
    # 定义计算动量的时间间隔
    periods = [1, 2, 3, 5, 10, 20]
    
    current_price_col = '收盘价_主力合约'
    
    for p in periods:
        if p < max_lag: # lag_p 列存在
            lag_col = f'{base_metric}_lag_{p}'
            
            # 动量: 当前价格 - N天前价格
            df[f'momentum_{p}'] = df[current_price_col] - df[lag_col]
            
            # 变化率: (当前价格 - N天前价格) / N天前价格
            # 加上一个很小的数防止除以0
            df[f'roc_{p}'] = (df[current_price_col] - df[lag_col]) / (df[lag_col] + 1e-9)
    return df.copy()

def generate_rsi_features(df: pd.DataFrame, max_lag: int = 60) -> pd.DataFrame:
    """
    计算RSI (相对强弱指数) 特征。
    """
    base_metric = '收盘价_主力合约'
    windows = [14, 20, 30] # RSI常用窗口期

    # 包含当天价格的完整序列 (T, T-1, T-2, ...)
    full_price_series_cols = [base_metric] + get_lag_cols(base_metric, max_lag)
    
    for w in windows:
        if w < len(full_price_series_cols):
            window_cols = full_price_series_cols[:w+1]
            
            # 计算每天的价格变化
            # 注意我们的列是 [P_t, P_{t-1}, ...], diff后是 [P_t - P_{t-1}, P_{t-1} - P_{t-2}, ...]
            price_diff = df[window_cols].diff(axis=1, periods=-1).iloc[:, :-1]
            
            # 分离上涨和下跌
            gain = price_diff.where(price_diff > 0, 0)
            loss = -price_diff.where(price_diff < 0, 0)
            
            # 计算平均上涨和平均下跌
            avg_gain = gain.mean(axis=1)
            avg_loss = loss.mean(axis=1)
            
            # 计算RS和RSI
            rs = avg_gain / (avg_loss + 1e-9)
            df[f'rsi_{w}'] = 100 - (100 / (1 + rs))
    
    return df

def generate_bollinger_bands_features(df: pd.DataFrame, max_lag: int = 60) -> pd.DataFrame:
    """
    计算布林带 (Bollinger Bands) 相关特征。
    """
    base_metric = '收盘价_主力合约'
    windows = [20, 30] # 常用窗口期
    
    price_cols = get_lag_cols(base_metric, max_lag)
    current_price_col = base_metric
    
    for w in windows:
        if w <= max_lag:
            window_cols = price_cols[:w]
            
            # 中轨: N日移动平均线
            middle_band = df[window_cols].mean(axis=1)
            
            # 标准差
            std_dev = df[window_cols].std(axis=1)
            
            # 上下轨
            upper_band = middle_band + 2 * std_dev
            lower_band = middle_band - 2 * std_dev
            
            # 特征1: 布林带宽度 (衡量波动性)
            df[f'bb_width_{w}'] = (upper_band - lower_band) / (middle_band + 1e-9)
            
            # 特征2: 当前价格在布林带中的位置 (%B)
            df[f'bb_percent_b_{w}'] = (df[current_price_col] - lower_band) / (upper_band - lower_band + 1e-9)
            
    return df

def generate_contract_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成主力、次大、第三大合约之间的价差、比率特征。
    """
    # --- 价格 Spread & Ratio ---
    df['spread_close_main_vs_sub'] = df['收盘价_主力合约'] - df['收盘价_次大合约']
    df['ratio_close_main_vs_sub'] = df['收盘价_主力合约'] / (df['收盘价_次大合约'] + 1e-9)
    
    df['spread_close_sub_vs_third'] = df['收盘价_次大合约'] - df['收盘价_第三大合约']
    df['ratio_close_sub_vs_third'] = df['收盘价_次大合约'] / (df['收盘价_第三大合约'] + 1e-9)
    
    # --- 成交量 Ratio ---
    df['ratio_volume_main_vs_sub'] = df['成交量_主力合约'] / (df['成交量_次大合约'] + 1e-9)
    df['ratio_volume_main_vs_total'] = df['成交量_主力合约'] / (df['成交量_主力合约'] + df['成交量_次大合约'] + df['成交量_第三大合约'] + 1e-9)

    # --- 持仓量 Ratio ---
    df['ratio_oi_main_vs_sub'] = df['持仓量_主力合约'] / (df['持仓量_次大合约'] + 1e-9)
    df['ratio_oi_main_vs_total'] = df['持仓量_主力合约'] / (df['持仓量_主力合约'] + df['持仓量_次大合约'] + df['持仓量_第三大合约'] + 1e-9)
    
    return df

# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_GLOBAL: Dict[str, Callable] = {
    # 1 魔法列处理
    "add_date_info": add_date_info,
    'add_volume_change': add_volume_change,
    'add_intraday_volatility': add_intraday_volatility,
    "generate_rolling_stats_features": generate_rolling_stats_features,
    'generate_momentum_features': generate_momentum_features,
    'generate_rsi_features': generate_rsi_features,
    'generate_bollinger_bands_features': generate_bollinger_bands_features,
    'generate_contract_spread_features': generate_contract_spread_features,

    # 2 非数值列信息编码，目前不需要，因为所有列都是数值列，后续有了再更新
    # "label_encode": label_encode,

    # 3 缺失值处理
    "fill_with_stat": fill_with_stat,
    "fill_with_value": fill_with_value,
    "handle_missing_values_for_oil_data": handle_missing_values_for_oil_data,    # 定制化缺失值处理

    # 选择模型训练的列
    "update_used_fields": update_used_fields,
}