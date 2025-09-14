import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging # 导入logging模块

# Scikit-learn a necessary tools
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, RFE
from src.evaluation.evaluator import calculate_pps

# --- 日志配置 ---
# 配置日志记录，将信息输出到 ./debug.log 文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='./debug.log',
    filemode='w'  # 'w' 表示每次运行都覆盖旧日志, 如果想追加请使用 'a'
)
log = logging.getLogger()

# --- 3. 粗筛选阶段 ---
def run_coarse_filtering(X_train, y_train, mi_keep_ratio=0.7, corr_threshold=0.9):
    """执行粗筛选流程"""
    # 移除高度相关特征
    mi_scores = pd.Series(
        mutual_info_regression(X_train, y_train, random_state=42),
        index=X_train.columns,
        name="MI_Score"
    )

    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(col, row, upper_tri.loc[row, col]) for col in upper_tri.columns for row in upper_tri.index if upper_tri.loc[row, col] > corr_threshold]

    # 根据互信息分数决定从每对中移除哪一个
    to_drop = set()
    for col1, col2, p_score in high_corr_pairs:
        # 如果其中一个已经在待移除列表里，则跳过，避免重复判断
        if col1 in to_drop or col2 in to_drop:
            continue

        mi_col1 = mi_scores.get(col1, 0)
        mi_col2 = mi_scores.get(col2, 0)
        if mi_col1 < mi_col2:
            to_drop.add(col1)
            log.info(f"高相关对: ('{col1}', '{col2}'). 决策: 移除 '{col1}' (MI: {mi_col1:.4f} < {mi_col2:.4f}), (P-score: {p_score})")
        else:
            to_drop.add(col2)
            log.info(f"高相关对: ('{col1}', '{col2}'). 决策: 移除 '{col2}' (MI: {mi_col2:.4f} <= {mi_col1:.4f}), (P-score: {p_score})")

    selected_features_corr = X_train.columns.drop(list(to_drop))
    log.info(f"移除高相关特征后，剩余特征: {len(selected_features_corr)}")

    # 3: 基于互信息筛选
    # 从上一步幸存的特征中，根据MI分数进行最终筛选
    mi_scores_filtered = mi_scores[selected_features_corr].sort_values(ascending=False)
    n_keep_mi = int(len(mi_scores_filtered) * mi_keep_ratio)
    final_selected_features = mi_scores_filtered.head(n_keep_mi).index

    log.info(f"互信息筛选后，最终剩余特征: {len(final_selected_features)}")
    log.info(f"最终选择的特征: {final_selected_features.to_list()}")

    # 应用最终的特征列表到训练集和测试集
    # X_train_final = X_train[final_selected_features]
    # 注意：原始代码中 X_test 未定义，这里假设它在调用函数的上下文中存在
    # 为了让函数能独立运行，这里暂时注释掉，您在使用时需要确保 X_test 是可访问的
    # X_test_final = X_test[final_selected_features]
    # return X_train_final, X_test_final

    # 根据您提供的代码片段，此函数似乎只返回了X_train_final和X_test_final
    # 但在main流程中，它还被期望返回 coarse_features。
    # 为了匹配主流程的调用，我将返回修改为:
    return X_train[final_selected_features], X_test_main[final_selected_features], final_selected_features


# --- 4. 高级降噪：Null Importance ---
def run_null_importance(X, y, features, n_runs=100):
    """执行Null Importance来识别真正有意义的特征"""
    log.info("--- 开始执行高级降噪 (Null Importance) ---")
    
    # 训练一个基础XGBoost模型
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, tree_method='hist', device='cuda')
    
    # 1. 计算真实重要性
    base_model.fit(X[features], y)
    real_importances = pd.Series(base_model.feature_importances_, index=features, name="real_importance")
    
    # 2. 计算噪音重要性分布
    null_importances = pd.DataFrame(index=features)
    y_shuffled = y.copy()
    
    for i in tqdm(range(n_runs), desc="Null Importance Runs"):
        np.random.shuffle(y_shuffled.values)
        base_model.fit(X[features], y_shuffled)
        null_importances[f'run_{i}'] = base_model.feature_importances_
        
    # 3. 筛选特征
    thresholds = null_importances.quantile(0.95, axis=1)
    significant_features = real_importances[real_importances > thresholds].index.tolist()
    
    log.info(f"Null Importance筛选后剩余特征: {len(significant_features)}")
    log.info(f"Null Importance筛选后剩余特征: {significant_features}")
    log.info("--- 高级降噪完成 ---")
    
    return significant_features

# --- 5. 最终优化方法A：RFE循环验证 ---
def find_optimal_features_rfe(X, y, features, n_to_test):
    """使用RFE和交叉验证寻找最优特征数量"""
    log.info("--- 开始最终优化方法A (RFE 循环验证) ---")
    
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, tree_method='hist', device='cuda')

    for n in tqdm(n_to_test, desc="Testing N features with RFE"):
        fold_scores = []
        for train_idx, val_idx in tscv.split(X[features]):
            X_train_fold, X_val_fold = X[features].iloc[train_idx], X[features].iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            rfe = RFE(estimator=base_model, n_features_to_select=n, step=0.1) # step=0.1表示每次移除10%
            rfe.fit(X_train_fold, y_train_fold)
            
            selected_features = X_train_fold.columns[rfe.support_]
            
            final_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, tree_method='hist', device='cuda')
            final_model.fit(X_train_fold[selected_features], y_train_fold)
            
            preds = final_model.predict(X_val_fold[selected_features])
            score = np.sqrt(calculate_pps(y_val_fold, preds))
            fold_scores.append(score)
            
        avg_score = np.mean(fold_scores)
        results.append({'n_features': n, 'avg_pps': avg_score})
        log.info(f"RFE with n={n}: Average pps = {avg_score:.4f}")

    results_df = pd.DataFrame(results)
    best_n = results_df.loc[results_df['avg_pps'].idxmin()]
    
    log.info(f"最优特征数量为: {int(best_n['n_features'])} (pps: {best_n['avg_pps']:.4f})")
    
    # 用找到的最佳N在全部训练集上最后运行一次RFE
    rfe_final = RFE(estimator=base_model, n_features_to_select=int(best_n['n_features']))
    rfe_final.fit(X[features], y)
    final_features_rfe = X[features].columns[rfe_final.support_].tolist()
    
    # 绘制性能曲线
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='n_features', y='avg_pps', marker='o')
    plt.title('RFE: Performance vs. Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Average Cross-Validated pps')
    plt.show()

    log.info("--- RFE 循环验证完成 ---")
    return final_features_rfe

# --- 7. 主流程执行 ---
if __name__ == '__main__':
    init_columns = []
    init_columns.extend(['close_rolling_mean_5', 'close_rolling_std_5', 'close_rolling_max_5', 'close_rolling_min_5', 'close_rolling_skew_5', 'close_rolling_kurt_5', 'close_rolling_mean_10', 'close_rolling_std_10', 'close_rolling_max_10', 'close_rolling_min_10', 'close_rolling_skew_10', 'close_rolling_kurt_10', 'close_rolling_mean_20', 'close_rolling_std_20', 'close_rolling_max_20', 'close_rolling_min_20', 'close_rolling_skew_20', 'close_rolling_kurt_20', 'close_rolling_mean_40', 'close_rolling_std_40', 'close_rolling_max_40', 'close_rolling_min_40', 'close_rolling_skew_40', 'close_rolling_kurt_40', 'close_rolling_mean_60', 'close_rolling_std_60', 'close_rolling_max_60', 'close_rolling_min_60', 'close_rolling_skew_60', 'close_rolling_kurt_60'])
    init_columns.extend(['momentum_1', 'roc_1', 'momentum_2', 'roc_2', 'momentum_3', 'roc_3', 'momentum_5', 'roc_5', 'momentum_10', 'roc_10', 'momentum_20', 'roc_20'])
    init_columns.extend(['rsi_14', 'rsi_20', 'rsi_30'])
    init_columns.extend(['bb_width_30', 'bb_width_20', 'bb_percent_b_20', 'bb_percent_b_30'])
    init_columns.extend(['spread_close_main_vs_sub', 'ratio_close_main_vs_sub', 'spread_close_sub_vs_third', 'ratio_close_sub_vs_third', 'ratio_volume_main_vs_sub', 'ratio_volume_main_vs_total', 'ratio_oi_main_vs_sub', 'ratio_oi_main_vs_total'])


    for pkl_path in ['cache_feature_engineering/sc.pkl', 'cache_feature_engineering/brent.pkl', 'cache_feature_engineering/wti.pkl']:
        for target_column in ['T_5', 'T_10', 'T_20']:
            log.info(f"当前处理{pkl_path}, {target_column}")
            # --- 1. 数据准备 ---
            df = pd.read_pickle(pkl_path)
            y = df[target_column]
            X = df.drop(columns=['T_5', 'T_10', 'T_20'])[init_columns]

            # --- 2. 划分训练集和最终测试集 ---
            # 严格按时间划分，最后的20%作为从未见过的测试集
            test_size = int(len(X) * 0.2)
            X_train_main, X_test_main = X[:-test_size], X[-test_size:]
            y_train_main, y_test_main = y[:-test_size], y[-test_size:]
            log.info(f"训练集大小: {X_train_main.shape}, 测试集大小: {X_test_main.shape}")
            
            # 阶段一：粗筛选
            # (注意: 我修正了 run_coarse_filtering 的返回值以匹配这里的用法)
            X_train_coarse, X_test_coarse, coarse_features = run_coarse_filtering(X_train_main, y_train_main)

            
            # 阶段二：高级降噪
            significant_features = run_null_importance(
                X_train_coarse, y_train_main, coarse_features
            )
            
            # 更新训练集和测试集以仅包含显著特征
            # X_train_sig = X_train_coarse[significant_features]
            # X_test_sig = X_test_coarse[significant_features]
            
            # 阶段三：最终优化
            # 方法A: RFE
            # n_to_test = [20, 40, 60, 80, 100, 125, 150] # 缩小测试范围，因为输入特征已减少
            # rfe_selected_features = find_optimal_features_rfe(
            #     X_train_sig, y_train_main, significant_features, n_to_test
            # )
            
            
            # # --- 8. 最终评估 ---
            # log.info("--- 开始最终评估 ---")
            
            # # 评估RFE选择的特征
            # final_model_rfe = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, tree_method='hist', device='cuda')
            # final_model_rfe.fit(X_train_main[rfe_selected_features], y_train_main)
            # preds_rfe = final_model_rfe.predict(X_test_main[rfe_selected_features])
            # pps_rfe = np.sqrt(calculate_pps(y_test_main, preds_rfe))
            # log.info(f"使用RFE选择的 {len(rfe_selected_features)} 个特征，在最终测试集上的pps: {pps_rfe:.4f}")