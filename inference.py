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
from copy import deepcopy


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def main(config_path: str, debug: bool):  

    # 1. 加载配置
    logging.info(f"--- [1/6] 加载配置 from {config_path} ---")
    cfg = load_config(config_path)
    cfg.base_data_dir = f"{args.input_path}/"

    # 设置随机种子
    set_random_seed(cfg.get("seed", 42))
    logging.info(f"配置加载成功: \n{pformat(cfg)}")

    # 2. 数据加载与整合、特征工程、预测、保存结果，遍历每个品种
    logging.info("--- [2/6] 开始数据加载与整合 ---")
    prediction_results = []
    # 品种名映射表
    product_code_map = {'sc': 'SC', 'brent': 'Brent', 'wti': 'WTI'}
    for product_name, product_path in cfg.data_loader.trade_data.items():
        # logging.info(f"处理品种: {product_name}")
        dfs = load_data(
            base_data_dir=cfg.base_data_dir,
            trade_data_path=str(product_path),
            fundamental_paths=cfg.data_loader.fundamental_data_paths
        )
        dfs = clean_data(dfs)
        df = preprocess_data(dfs, cfg.preprocess_config)
        logging.info(f"数据整合完成, shape: {df.shape}")
        # df.to_csv('1.csv')
        # df = pd.read_csv("1.csv")
        # df['日期'] = pd.to_datetime(df['日期'])
        # df.set_index('日期', inplace=True)

        for target_column in cfg.data_loader.target_columns:
            copy_df = df.copy()
            # 特征工程
            copy_df = feature_engineering_pipeline(copy_df, cfg[f"{product_name}_{target_column}_pipeline"])

            # 数据划分
            copy_df = copy_df.dropna(subset=[target_column])
            drop_cols = [x for x in ['T_5', 'T_10', 'T_20'] if x in copy_df.columns]
            X = copy_df.drop(columns=drop_cols)

            logging.info("--- [4/6] 开始构建预测数据集 ---")
            DATE = X.index
            X_pred = X.values
            logging.info(f"X_pred: {X_pred.shape}")
            
            # 5. 模型推理
            save_model_path = os.path.join(cfg.model_save_dir, cfg.experiment_name, product_name,  f'{target_column}.joblib')

            logging.info("--- 开始推理 ---")
            predictor = Predictor(save_model_path=save_model_path)
            y_pred = predictor.predict(X_pred)
            logging.info(f"推理完成, y_pred shape: {y_pred.shape}")

            # 6. 结果保存
            for idx in range(y_pred.shape[0]):
                date = DATE[idx]
                # 这里假设每个品种只输出自己的预测结果
                row = {
                    'date': date,
                    'product_code': product_code_map.get(product_name.lower(), product_name),
                    'target_horizon': target_column,
                    'predicted_price': y_pred[idx]
                }
                prediction_results.append(row)

    # 将结果列表转换为DataFrame
    results_df = pd.DataFrame(prediction_results)
    # 1. 日期格式化为YYYY-MM-DD，兼容datetime/date/字符串
    results_df['date'] = results_df['date'].apply(lambda x: pd.to_datetime(str(x)).strftime('%Y-%m-%d'))
    # 2. 排序：先按date，再按target_horizon(自定义顺序)，再按product_code
    target_horizon_order = {'T+5': 0, 'T+10': 1, 'T+20': 2}
    results_df['target_horizon_order'] = results_df['target_horizon'].map(target_horizon_order)
    results_df = results_df.sort_values(by=['date', 'target_horizon_order', 'product_code']).reset_index(drop=True)
    results_df = results_df.drop(columns=['target_horizon_order'])
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output.csv') 
    results_df.to_csv(output_path, index=False, encoding='utf-8', date_format='%Y-%m-%d')

    logging.info(f"预测结果已保存至: {output_path}")


        
if __name__ == '__main__':
    # 固定参数，直接写死
    class Args:
        config = '/configs/predict_config.yaml'
        debug = False
        input_path = '/app/input/' 
        output_path = '/app/output/'
    args = Args()
    main(args.config, args.debug)