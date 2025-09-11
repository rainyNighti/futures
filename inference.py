import os
import argparse
import logging
from pprint import pformat

from src.utils import set_random_seed
from src.utils.config import load_config
from src.data.data_loader import assemble_data
from src.features.preprocessing import execute_preprocessing_pipeline
from src.data.dataset import generate_predict_dataset
from src.modeling.trainer import ModelTrainer
from src.modeling.predictor import Predictor
from src.evaluation.evaluator import Evaluator
import os
import random
import numpy as np
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def main(config_path: str, debug: bool):  
    if debug:
        import debugpy;debugpy.listen(("localhost", 5678));print("Waiting for debugger attach...");debugpy.wait_for_client()

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
    for product_name in cfg.data_loader.trade_data.keys():
        logging.info(f"处理品种: {product_name}")
        df = assemble_data(
            base_data_dir=cfg.base_data_dir,
            trade_data_config=cfg.data_loader.trade_data,
            fundamental_paths=cfg.data_loader.fundamental_data_paths,
            product_name=product_name
        )
        logging.info(f"数据整合完成, shape: {df.shape}")

        # 3. 特征工程
        logging.info("--- [3/6] 开始执行特征工程流程 ---")
        df = execute_preprocessing_pipeline(df, cfg.preprocessing_pipeline, product_name=product_name)
        logging.info(f"特征工程完成, shape: {df.shape}")

        # 4. 预测数据集生成
        logging.info("--- [4/6] 开始构建预测数据集 ---")
        X_pred, DATE = generate_predict_dataset(df, cfg.dataset, product_name=product_name)
        logging.info(f"X_pred: {X_pred.shape}")
        
        # 5. 模型推理
        model_dir = os.path.join(cfg.model_save_dir, cfg.experiment_name, product_name)

        logging.info("--- 开始推理 ---")
        predictor = Predictor(model_dir=model_dir)
        y_pred = predictor.predict(X_pred)
        logging.info(f"推理完成, y_pred shape: {y_pred.shape}")

        # 6. 结果保存
        for idx in range(y_pred.shape[0]):
            date = DATE[idx]
            # 这里假设每个品种只输出自己的预测结果
            future_steps = [5, 10, 20]
            for i, step in enumerate(future_steps):
                # 预测值索引
                pred_idx = i
                predicted_price = y_pred[idx, pred_idx] if y_pred.ndim > 1 else y_pred[idx]
                row = {
                    'date': date,
                    'product_code': product_name,
                    'target_horizon': f"T+{step}",
                    'predicted_price': predicted_price
                }
                prediction_results.append(row)

    # 将结果列表转换为DataFrame并保存结果
    results_df = pd.DataFrame(prediction_results)
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output.csv') 
    results_df.to_csv(output_path, index=False, encoding='utf-8')

    logging.info(f"预测结果已保存至: {output_path}")


        
if __name__ == '__main__':
    # 固定参数，直接写死
    class Args:
        config = '/configs/predict_config.yaml'
        debug = False
        input_path = '/app/input/data/' 
        output_path = '/app/output'
    args = Args()
    main(args.config, args.debug)