from math import pi
import os
import argparse
import logging
from pprint import pformat
from collections import defaultdict

from src.utils.utils import set_random_seed, print_pretty_results
from src.utils.config import load_config
from src.data.data_loader import load_data
from src.data.data_clean import clean_data
from src.features.preprocessing import execute_preprocessing_pipeline
from src.data.dataset import split_dataset
from src.modeling.trainer import ModelTrainer
from src.modeling.predictor import Predictor
from src.evaluation.evaluator import calculate_pps
import os
import numpy as np
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_pipeline_config(pipeline_config, target_column: str):
    """ 将target_column动态注入到pipline config """
    for func in pipeline_config.trade_pipeline:
        if func.type == 'create_target_feature':
            func.target_column = target_column
    for func in pipeline_config.global_pipeline:
        if func.type == 'drop_all_nan_rows_and_y_nan_rows':
            func.target_column = target_column
    return pipeline_config

def main(config_path: str, debug: bool, extra_params: str):  
    if debug:
        import debugpy;debugpy.listen(("localhost", 5678));print("Waiting for debugger attach...");debugpy.wait_for_client()

    logging.info(f"加载配置 from {config_path}")
    cfg = load_config(config_path, extra_params)

    # 设置随机种子
    set_random_seed(cfg.get("seed", 42))
    logging.info(f"配置加载成功: \n{pformat(cfg)}")

    results = defaultdict(lambda: defaultdict(float))
    for product_name, product_path in tqdm(cfg.data_loader.trade_data.items(), desc='逐个处理交易所'):
        for target_column in cfg.data_loader.target_columns:    # 逐个y处理数据，训练模型
            # 1 加载所有csv数据
            dfs = load_data(
                base_data_dir=cfg.base_data_dir,
                trade_data_path=product_path,
                fundamental_paths=cfg.data_loader.fundamental_data_paths
            )

            # 2 数据清洗
            dfs = clean_data(dfs)

            # 3 特征工程
            pipeline_config = update_pipeline_config(cfg[f"{product_name}_{target_column}_pipeline"], target_column)
            df = execute_preprocessing_pipeline(dfs, pipeline_config, target_column)

            # 4 数据集划分
            X_train, X_test, y_train, y_test = split_dataset(df, cfg.test_split_ratio, target_column)
            logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

            # 5 模型训练
            save_model_path = os.path.join(cfg.model_save_dir, cfg.experiment_name, product_name, f'{target_column}.joblib')
            trainer = ModelTrainer(model_config=cfg.model, save_model_path=save_model_path)
            trainer.train(X_train, y_train, random_state=cfg.get("seed", 42))

            # 6 模型推理
            predictor = Predictor(save_model_path)
            y_pred = predictor.predict(X_test)
            
            # 7 性能评估
            results[product_name][target_column] = calculate_pps(y_test, y_pred)
        
    logging.info("\n--- 评测结果 ---")
    # 添加一个overall的平均分
    for target_column in cfg.data_loader.target_columns:
        total_score = 0
        for product_name, product_score in results.items():
            total_score += product_score[target_column]
        results['overall'][target_column] = total_score / 3
    final_score = np.mean([v for k, v in results['overall'].items()])   # 我们不设置权重了，final score就是三家产品平均值的平均值
    print_pretty_results(results, final_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行机器学习训练流程")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/base_config.yaml', 
        help='配置文件路径'
    )
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--extra_params', type=str, help='额外的配置参数，格式为key1=value1,key2=value2')
    args = parser.parse_args()
    main(args.config, args.debug, args.extra_params)