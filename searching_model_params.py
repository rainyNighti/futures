from math import pi
import os
import pandas as pd
import argparse
import logging
from pprint import pformat
from collections import defaultdict

from src.utils.utils import set_random_seed, print_pretty_results, write_score_to_csv
from src.utils.config import load_config
from src.data.data_loader import load_data
from src.data.data_clean import clean_data
from src.features.preprocessing import feature_engineering_pipeline, preprocess_data
from src.data.dataset import split_dataset
from src.modeling.trainer import ModelTrainer
from src.modeling.predictor import Predictor
from src.evaluation.evaluator import calculate_pps
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
# import cupy as cp

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path: str, debug: bool, extra_params: str, force_reprocess: bool):  
    if debug:
        import debugpy;debugpy.listen(("localhost", 5678));print("Waiting for debugger attach...");debugpy.wait_for_client()

    logging.info(f"加载配置 from {config_path}")
    cfg = load_config(config_path, extra_params)
    logging.info(f"配置加载成功: \n{pformat(cfg)}")
    

    tscv = TimeSeriesSplit(n_splits=10)     # 一共1607条数据，10折，1折是160条

    booster_choices = ['gbtree', 'dart', 'gblinear']
    gamma_choices = [0, 1, 5, 10]
    max_depth_choices = [3, 5, 7, 9]
    learning_rate_choices = [0.01, 0.05, 0.1, 0.2]
    min_child_weight_choices = [0.5, 1, 2, 5]
    subsample_choices = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    sampling_method_choices = ['uniform', 'gradient_based']
    lambda_choices = [0, 1, 2, 5]
    grow_policy_choices = ['depthwise', 'lossguide']
    max_bin_choices = [64, 128, 256, 512]
    num_parallel_tree_choices = [1, 2, 3]
    tree_method_choices = ['auto', 'exact', 'approx', 'hist']

    search_params = {
        'booster': booster_choices,
        'gamma': gamma_choices,
        'max_depth': max_depth_choices,
        'learning_rate': learning_rate_choices,
        'min_child_weight': min_child_weight_choices,
        'subsample': subsample_choices,
        'sampling_method': sampling_method_choices,
        'lambda': lambda_choices,
        'grow_policy': grow_policy_choices,
        'max_bin': max_bin_choices,
        'num_parallel_tree': num_parallel_tree_choices,
        'tree_method': tree_method_choices
    }

    def train_model_with_cfg(cfg, param, value):
        this_cfg = deepcopy(cfg)
        this_cfg.model.params[param] = value
        pps_values = []

        for idx, (train_index, test_index) in enumerate(tscv.split(X)):
            if len(train_index) >= 800:   # 至少有800条数据再开始训练 
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                # X_train, X_test, y_train, y_test = cp.array(X_train), cp.array(X_test), cp.array(y_train), cp.array(y_test)

                # 模型训练
                save_model_path = os.path.join(this_cfg.model_save_dir, this_cfg.experiment_name, product_name, str(idx), f'{target_column}.joblib')
                trainer = ModelTrainer(model_config=this_cfg.model, save_model_path=save_model_path)
                trainer.train(X_train, y_train, random_state=this_cfg.get("seed", 42))

                # 模型推理
                predictor = Predictor(save_model_path)
                y_pred = predictor.predict(X_test)
                
                # 7 性能评估
                pps = calculate_pps(y_test, y_pred)
                pps_values.append(pps)
        result_value = np.mean(pps_values) if pps_values else float('inf')
        print(f"参数 {param}={value} 的PPS值: {result_value}")
        return result_value

    all_best_params = {}

    for product_name, product_path in tqdm(cfg.data_loader.trade_data.items()):
        # 数据读取与合并
        cache_file = f"./cache"
        os.makedirs(cache_file, exist_ok=True)
        cache_file = os.path.join(cache_file, f'{product_name}.csv')
        if os.path.exists(cache_file) and not force_reprocess:
            logging.info(f"跳过 {product_name} 的缓存文件已存在: {cache_file}")
            df = pd.read_csv(cache_file)
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
        else:
            dfs = load_data(
                base_data_dir=cfg.base_data_dir,
                trade_data_path=product_path,
                fundamental_paths=cfg.data_loader.fundamental_data_paths
            )
            dfs = clean_data(dfs)
            df = preprocess_data(dfs, cfg.preprocess_config)
            df.to_csv(cache_file)
        product_best_params = {}

        for target_column in cfg.data_loader.target_columns:
            copy_df = df.copy()
            # 特征工程
            copy_df = feature_engineering_pipeline(copy_df, cfg[f"{product_name}_{target_column}_pipeline"])

            # 数据划分
            copy_df = copy_df.dropna(subset=[target_column])
            y = copy_df[target_column]
            X = copy_df.drop(columns=cfg.target_columns)


            target_column_best_params = {}
            for param, values in search_params.items():
                best_pps = float('inf')
                best_value = None
                for value in values:
                    mean_pps = train_model_with_cfg(cfg, param, value)
                    if mean_pps < best_pps:
                        best_pps = mean_pps
                        best_value = value
                target_column_best_params[param] = best_value
                logging.info(f"当前product {product_name}预测{target_column}的最佳模型参数 {param}: {best_value} with PPS: {best_pps}")
            logging.info(f"最终 {product_name} 预测 {target_column} 的最佳参数组合: {target_column_best_params}")
            product_best_params[target_column] = target_column_best_params
        all_best_params[product_name] = product_best_params
        logging.info(f"产品 {product_name} 的所有目标列最佳参数: {product_best_params}")
    logging.info(f"所有产品的最佳参数汇总: {all_best_params}")
    import json
    with open('best_model_params.json', 'w') as f:
        json.dump(all_best_params, f, indent=4)

if __name__ == '__main__':
    set_random_seed(42)

    parser = argparse.ArgumentParser(description="运行机器学习训练流程")
    parser.add_argument('config', type=str)
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--force_reprocess', action='store_true', help='强制覆盖生成的data cache')
    parser.add_argument('--extra_params', type=str, help='额外的配置参数，格式为key1=value1,key2=value2')
    args = parser.parse_args()
    main(args.config, args.debug, args.extra_params, args.force_reprocess)