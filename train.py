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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path: str, debug: bool, extra_params: str, force_reprocess: bool):  
    if debug:
        import debugpy;debugpy.listen(("localhost", 5678));print("Waiting for debugger attach...");debugpy.wait_for_client()

    tqdm.write(f"加载配置 from {config_path}")
    cfg = load_config(config_path, extra_params)
    tqdm.write(f"配置加载成功: \n{pformat(cfg)}")
    
    # 不需要num seed，使用TimeSeriesSplit
    results = defaultdict(lambda: defaultdict(dict))
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

        for target_column in cfg.data_loader.target_columns:
            copy_df = df.copy()
            # 特征工程
            copy_df = feature_engineering_pipeline(copy_df, cfg[f"{product_name}_{target_column}_pipeline"])

            # 数据划分
            copy_df = copy_df.dropna(subset=[target_column])
            y = copy_df[target_column]
            X = copy_df.drop(columns=cfg.target_columns)
            tscv = TimeSeriesSplit(n_splits=10)     # 一共1607条数据，10折，1折是160条

            pps_scores = []
            r2_scores = []

            for idx, (train_index, test_index) in enumerate(tscv.split(X)):
                if len(train_index) >= 800:   # 至少有800条数据再开始训练 
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # 模型训练
                    save_model_path = os.path.join(cfg.model_save_dir, cfg.experiment_name, product_name, str(idx), f'{target_column}.joblib')
                    if cfg.model.type == "aemlp":
                        from src.modeling.aemlp import AE_MLP_Model, loss_fn
                        feat_dim = X_train.shape[1]
                        target_dim = 1
                        model = AE_MLP_Model(
                            feat_dim=feat_dim,
                            target_dim=target_dim,
                            ae_hidden=cfg.model.ae_hidden,
                            ae_code=cfg.model.ae_code,
                            mlp_hidden=cfg.model.mlp_hidden,
                            dropout=cfg.model.dropout,
                            noise_std=cfg.model.noise_std
                        )
                        import torch
                        import torch.optim as optim
                        # 转换数据为张量
                        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
                        x_mean, x_std = X_train_tensor.mean(0), X_train_tensor.std(0)
                        X_train_tensor = (X_train_tensor - x_mean) / (x_std + 1e-6)  # 标准化
                        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
                        y_mean, y_std = y_train_tensor.mean(0), y_train_tensor.std(0)
                        y_train_tensor = (y_train_tensor - y_mean) / (y_std + 1e-6)  # 标准化
                        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                        X_test_tensor = (X_test_tensor - x_mean) / (x_std + 1e-6)  # 标准化
                        optimizer = optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
                        # 训练模型
                        model.train()
                        best_pps = -np.inf
                        final_pps, final_r2 = -np.inf, -np.inf
                        for epoch in range(cfg.model.recon_epochs + cfg.model.mlp_epochs):
                            if epoch < cfg.model.recon_epochs:
                                alpha = 1.0  # 只训练自编码器部分
                            else:
                                alpha = 0.0  # 只训练MLP部分
                            optimizer.zero_grad()
                            out, recon = model(X_train_tensor, y_train_tensor)
                            loss, bce = loss_fn(out, y_train_tensor, recon, X_train_tensor, alpha=alpha)
                            loss.backward()
                            optimizer.step()
                            # 模型推理
                            model.eval()
                            with torch.no_grad():
                                y_pred, _ = model(X_test_tensor, is_inference=True)
                            y_pred = (y_pred * (y_std + 1e-6) + y_mean).numpy().squeeze()
                            y_test_np = y_test.values
                            pps = calculate_pps(y_test_np, y_pred)
                            r2 = r2_score(y_test_np, y_pred)
                            if epoch < cfg.model.recon_epochs:
                                tqdm.write(f"Recon loss: {loss.item():.4f}, BCE: {bce.item():.4f}, PPS: {pps:.4f}, R2: {r2:.4f} at epoch {epoch+1}/{cfg.model.recon_epochs}")
                            else:
                                tqdm.write(f"MLP loss: {loss.item():.4f}, BCE: {bce.item():.4f}, PPS: {pps:.4f}, R2: {r2:.4f} at epoch {epoch+1-cfg.model.recon_epochs}/{cfg.model.mlp_epochs}")
                                if pps > best_pps:
                                    best_pps = pps
                                    final_pps, final_r2 = pps, r2
                                    # 保存模型
                                    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
                                    torch.save(model.state_dict(), save_model_path)
                                else:
                                    tqdm.write("Early stopping as PPS did not improve.")
                                    break
                            model.train()
                            
                        pps_scores.append(final_pps)
                        r2_scores.append(final_r2)
                    else:
                        trainer = ModelTrainer(model_config=cfg.model, save_model_path=save_model_path)
                        trainer.train(X_train, y_train, random_state=cfg.get("seed", 42))

                        # 模型推理
                        predictor = Predictor(save_model_path)
                        y_pred = predictor.predict(X_test)
                    
                        # 7 性能评估
                        pps_scores.append(calculate_pps(y_test, y_pred))
                        r2_scores.append(r2_score(y_test, y_pred))

            results[product_name][target_column] = {
                'pps': np.mean(pps_scores),
                'r2': np.mean(r2_scores)
            }
            
    # 结果汇总与平均
    tqdm.write("\n--- 评测结果 ---")
    copy_dict = deepcopy(results)
    for target_column in cfg.data_loader.target_columns:
        total_score = 0
        total_score_r2 = 0
        for product_name, product_score in results.items():
            total_score += product_score[target_column]['pps']
            total_score_r2 += product_score[target_column]['r2']
        copy_dict['overall'][target_column] = total_score / 3
        copy_dict['r2'][target_column] = total_score_r2 / 3
    # 对所有seed的所有字段结果做平均
    final_score = np.mean(list(copy_dict['overall'].values()))
    # 打印所有seed的结果和平均结果
    print_pretty_results(copy_dict, final_score)
    write_score_to_csv(copy_dict, final_score, os.path.join(cfg.model_save_dir, cfg.experiment_name))

if __name__ == '__main__':
    set_random_seed(42)

    parser = argparse.ArgumentParser(description="运行机器学习训练流程")
    parser.add_argument('config', type=str)
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--force_reprocess', action='store_true', help='强制覆盖生成的data cache')
    parser.add_argument('--extra_params', type=str, help='额外的配置参数，格式为key1=value1,key2=value2')
    args = parser.parse_args()
    main(args.config, args.debug, args.extra_params, args.force_reprocess)