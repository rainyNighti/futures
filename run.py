import os
import argparse
import logging
from pprint import pformat

from src.utils import set_random_seed
from src.utils.config import load_config
from src.data.data_loader import assemble_data
from src.features.preprocessing import execute_preprocessing_pipeline
from src.data.dataset import create_and_split_supervised_dataset
from src.modeling.trainer import ModelTrainer
from src.modeling.predictor import Predictor
from src.evaluation.evaluator import Evaluator
import os
import random
import numpy as np


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path: str):  
    import debugpy;debugpy.listen(("localhost", 5678));print("Waiting for debugger attach...");debugpy.wait_for_client()

    # 1. 加载配置
    logging.info(f"--- [1/6] 加载配置 from {config_path} ---")
    cfg = load_config(config_path)

    # 设置随机种子
    set_random_seed(cfg.get("seed", 42))
    logging.info(f"配置加载成功: \n{pformat(cfg)}")

    # 2. 数据加载与整合
    logging.info("--- [2/6] 开始数据加载与整合 ---")
    df = assemble_data(
        base_data_dir=cfg.base_data_dir,
        trade_data_config=cfg.data_loader.trade_data,
        fundamental_paths=cfg.data_loader.fundamental_data_paths
    )
    logging.info(f"数据整合完成, shape: {df.shape}")

    # 3. 特征工程
    logging.info("--- [3/6] 开始执行特征工程流程 ---")
    df = execute_preprocessing_pipeline(df, cfg.preprocessing_pipeline)
    logging.info(f"特征工程完成, shape: {df.shape}")

    # 4. 数据集构建与划分
    logging.info("--- [4/6] 开始构建监督学习数据集 ---")
    X_train, y_train, X_test, y_test = create_and_split_supervised_dataset(df, cfg.dataset)
    logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 5. 模型训练与推理
    model_dir = os.path.join(cfg.model_save_dir, cfg.experiment_name)
    
    logging.info(f"--- [5/6] 开始模型训练，模型将保存在: {model_dir} ---")
    trainer = ModelTrainer(model_config=cfg.model, model_dir=model_dir)
    trainer.train(X_train, y_train)
    logging.info("模型训练完成。")

    logging.info("--- 开始推理 ---")
    predictor = Predictor(model_dir=model_dir)
    y_pred = predictor.predict(X_test)
    logging.info(f"推理完成, y_pred shape: {y_pred.shape}")
    
    # 6. 性能评估
    logging.info("--- [6/6] 开始性能评估 ---")
    evaluator = Evaluator(cfg.evaluation, cfg.dataset)
    scores = evaluator.calculate_final_score(y_test, y_pred)
    
    logging.info("\n--- 评测结果 ---")
    for key, value in scores.items():
        logging.info(f"{key}: {value:.6f}")
    logging.info("--- 程序执行完毕 ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行机器学习训练流程")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/base_config.yaml', 
        help='配置文件路径'
    )
    args = parser.parse_args()
    main(args.config)