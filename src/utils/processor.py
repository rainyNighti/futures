from typing import Dict
import pandas as pd
import numpy as np

def merge_with_lags_multi_freq(base_df: pd.DataFrame, 
                               df_to_merge: pd.DataFrame, 
                               freq_map: Dict[str, str], 
                               n_lags_m: int = 4,      # 这里都默认提取1年的数据
                               n_lags_w: int = 8, 
                               n_lags_d: int = 60,
                               n_lags_q: int = 2, 
                               n_lags_y: int = 2):

    """
    将一个时间序列DataFrame（df_to_merge）中的数据按不同的时间频率（年、季、月、周、日）
    以滞后特征（lags）的形式合并到基础DataFrame（base_df）中。
    以数据2025.8.2举例：
    - 对于年更数据：找当年(2025.1.1-2025.8.2)，前一年(2024.1.1-2024.12.31)，前前一年...的数据
    - 对于季更数据：找当季(2025.7.1-2025.8.2)，前一季(2025.4.1-2025.6.30)，前前一季...的数据
    - 对于月更数据：找当月(2025.8.1-2025.8.2)，前一月(2025.7.1-2025.7.31)，前前一月...的数据
    - 对于周更数据：找当周(周一至当天)，前一周(上周一至上周日)，前前一周...的数据
    - 对于日更数据：找当天，前一天，前前一天...的数据

    参数:
    - base_df (pd.DataFrame): 基础DataFrame，索引必须是唯一的pd.DatetimeIndex。
    - df_to_merge (pd.DataFrame): 待合并的特征来源DataFrame，索引必须是唯一的pd.DatetimeIndex。
    - freq_map (dict): 一个字典，键为df_to_merge中的列名，值为其更新频率 ('Y', 'Q', 'M', 'W', 'D')。
    - n_lags_m (int): 月度数据的滞后阶数。
    - n_lags_w (int): 周度数据的滞后阶数。
    - n_lags_d (int): 日度数据的滞后阶数。
    - n_lags_q (int): 季度数据的滞后阶数。
    - n_lags_y (int): 年度数据的滞后阶数。

    返回:
    - pd.DataFrame: 合并了所有滞后特征的最终DataFrame。
    """
    # --- 1. 输入验证与预处理 ---
    if not isinstance(base_df.index, pd.DatetimeIndex) or not isinstance(df_to_merge.index, pd.DatetimeIndex):
        raise TypeError("两个DataFrame的索引都必须是DatetimeIndex类型。")
    if not base_df.index.is_unique or not df_to_merge.index.is_unique:
        raise ValueError("两个DataFrame的索引都必须是唯一的。")

    # 为了高效查找，对待合并的df进行排序
    df_to_merge = df_to_merge.sort_index()

    # 根据freq_map拆分df_to_merge
    cols_y = [col for col, freq in freq_map.items() if freq == 'Y']
    cols_q = [col for col, freq in freq_map.items() if freq == 'Q']
    cols_m = [col for col, freq in freq_map.items() if freq == 'M']
    cols_w = [col for col, freq in freq_map.items() if freq == 'W']
    cols_d = [col for col, freq in freq_map.items() if freq == 'D']

    yearly_df = df_to_merge[cols_y].dropna(how='all') if cols_y else pd.DataFrame()
    quarterly_df = df_to_merge[cols_q].dropna(how='all') if cols_q else pd.DataFrame()
    monthly_df = df_to_merge[cols_m].dropna(how='all') if cols_m else pd.DataFrame()
    weekly_df = df_to_merge[cols_w].dropna(how='all') if cols_w else pd.DataFrame()
    daily_df = df_to_merge[cols_d].dropna(how='all') if cols_d else pd.DataFrame()

    # 将索引缓存为集合，以便进行更快的'in'检查 (主要对日度数据有效)
    daily_df_index_set = set(daily_df.index)

    # --- 2. 核心循环与特征计算 ---
    all_new_data = [] # 存储每一行生成的新特征

    for current_date in base_df.index:
        row_data = {}
        # --- 处理年度数据 ---
        if not yearly_df.empty and n_lags_y > 0:
            for i in range(n_lags_y):
                if i == 0:
                    start_date = current_date.replace(month=1, day=1)
                    end_date = current_date
                else:
                    target_year_date = current_date - pd.DateOffset(years=i)
                    start_date = target_year_date.replace(month=1, day=1)
                    end_date = start_date + pd.offsets.YearEnd(1)
                
                period_data = yearly_df.loc[start_date:end_date]
                
                if not period_data.empty:
                    latest_row = period_data.iloc[-1]
                    for col in yearly_df.columns:
                        row_data[f'{col}_Y_lag_{i+1}'] = latest_row[col]
                else:
                    for col in yearly_df.columns:
                        row_data[f'{col}_Y_lag_{i+1}'] = np.nan

        # --- 处理季度数据 ---
        if not quarterly_df.empty and n_lags_q > 0:
            for i in range(n_lags_q):
                if i == 0:
                    quarter_start_month = (current_date.quarter - 1) * 3 + 1
                    start_date = current_date.replace(month=quarter_start_month, day=1)
                    end_date = current_date
                else:
                    target_quarter_date = current_date - pd.DateOffset(months=i*3) 
                    quarter_start_month = (target_quarter_date.quarter - 1) * 3 + 1
                    start_date = target_quarter_date.replace(month=quarter_start_month, day=1)
                    end_date = start_date + pd.offsets.QuarterEnd(1)
                
                period_data = quarterly_df.loc[start_date:end_date]
                
                if not period_data.empty:
                    latest_row = period_data.iloc[-1]
                    for col in quarterly_df.columns:
                        row_data[f'{col}_Q_lag_{i+1}'] = latest_row[col]
                else:
                    for col in quarterly_df.columns:
                        row_data[f'{col}_Q_lag_{i+1}'] = np.nan

        # --- 处理月度数据 ---
        if not monthly_df.empty and n_lags_m > 0:
            for i in range(n_lags_m):
                if i == 0:
                    start_date = current_date.replace(day=1)
                    end_date = current_date
                else:
                    target_month_date = current_date - pd.DateOffset(months=i)
                    start_date = target_month_date.replace(day=1)
                    end_date = start_date + pd.offsets.MonthEnd(1)
                
                # 在时间窗口内查找数据
                period_data = monthly_df.loc[start_date:end_date]
                
                if not period_data.empty:
                    latest_row = period_data.iloc[-1] # 获取最后一行数据
                    for col in monthly_df.columns:
                        row_data[f'{col}_M_lag_{i+1}'] = latest_row[col]
                else:
                    for col in monthly_df.columns:
                        row_data[f'{col}_M_lag_{i+1}'] = np.nan

        # --- 处理周度数据 ---
        if not weekly_df.empty and n_lags_w > 0:
            start_of_current_week = current_date - pd.to_timedelta(current_date.dayofweek, unit='D')
            for i in range(n_lags_w):
                if i == 0:
                    start_date = start_of_current_week
                    end_date = current_date
                else:
                    end_of_period_week = start_of_current_week - pd.to_timedelta(1 + (i-1)*7, unit='D')
                    start_date = end_of_period_week - pd.to_timedelta(6, unit='D')
                    end_date = end_of_period_week
                
                period_data = weekly_df.loc[start_date:end_date]
                
                if not period_data.empty:
                    latest_row = period_data.iloc[-1]
                    for col in weekly_df.columns:
                        row_data[f'{col}_W_lag_{i+1}'] = latest_row[col]
                else:
                    for col in weekly_df.columns:
                        row_data[f'{col}_W_lag_{i+1}'] = np.nan

        # --- 处理日度数据 ---
        if not daily_df.empty and n_lags_d > 0:
            for i in range(n_lags_d):
                target_date = current_date - pd.to_timedelta(i, unit='D')
                
                if target_date in daily_df_index_set:
                    latest_row = daily_df.loc[target_date]
                    for col in daily_df.columns:
                        row_data[f'{col}_D_lag_{i+1}'] = latest_row[col]
                else:
                    for col in daily_df.columns:
                        row_data[f'{col}_D_lag_{i+1}'] = np.nan
        
        all_new_data.append(row_data)

    # --- 3. 合并结果 ---
    new_features_df = pd.DataFrame(all_new_data, index=base_df.index)
    final_df = pd.concat([base_df, new_features_df], axis=1)

    return final_df