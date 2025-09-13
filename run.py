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
from src.features.preprocessing import execute_preprocessing_pipeline
from src.data.dataset import split_dataset
from src.modeling.trainer import ModelTrainer
from src.modeling.predictor import Predictor
from src.evaluation.evaluator import calculate_pps
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

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
    logging.info(f"配置加载成功: \n{pformat(cfg)}")

    num_seeds = cfg.get("num_seeds", None)
    if num_seeds is not None:
        try:
            num_seeds = int(num_seeds)
        except Exception:
            num_seeds = None

    seeds_to_run = [cfg.get('seed', 42)] if not num_seeds else list(range(num_seeds))
    results = defaultdict(lambda: defaultdict(float))
    for product_name, product_path in tqdm(cfg.data_loader.trade_data.items()):
        for target_column in cfg.data_loader.target_columns:
            # cache is independent of seed
            os.makedirs('cache', exist_ok=True)
            cache_file = f"cache/{product_name}_{target_column}.pkl"
            if os.path.exists(cache_file) and not cfg.get("force_reprocess", False):
                logging.info(f"跳过 {product_name} 的 {target_column}，缓存文件已存在: {cache_file}")
                df = pd.read_pickle(cache_file)
            else:
                dfs = load_data(
                    base_data_dir=cfg.base_data_dir,
                    trade_data_path=product_path,
                    fundamental_paths=cfg.data_loader.fundamental_data_paths
                )
                dfs = clean_data(dfs)
                pipeline_config = update_pipeline_config(cfg[f"{product_name}_{target_column}_pipeline"], target_column)
                df = execute_preprocessing_pipeline(dfs, pipeline_config, target_column)
                df.to_pickle(cache_file)
            scores = []
            r2_scores = []
            for seed in seeds_to_run:
                set_random_seed(seed)
                cfg["seed"] = seed
                X_train, X_test, y_train, y_test = split_dataset(df, cfg.test_split_ratio, target_column)
                logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
                logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
                save_model_path = os.path.join(cfg.model_save_dir, cfg.experiment_name, product_name, f'{target_column}_seed{seed}.joblib')
                trainer = ModelTrainer(model_config=cfg.model, save_model_path=save_model_path)
                model = trainer.train(X_train, y_train, random_state=seed)
                save_importance_path = os.path.join(cfg.model_save_dir, cfg.experiment_name, product_name, f'{target_column}_seed{seed}_important.csv')
                is_inference = False
                if cfg.model.type == 'xgboost' and not is_inference:
                    importances = model.feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': df.drop(columns=target_column).columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    importance_gain = model.get_booster().get_score(importance_type='gain')
                    feature_names = list(df.drop(columns=target_column).columns)
                    feature_map = {f"f{i}": name for i, name in enumerate(feature_names)}
                    importance_gain_df = pd.DataFrame({
                        'Feature': [feature_map.get(f, f) for f in importance_gain.keys()],
                        'Gain': importance_gain.values()
                    }).sort_values('Gain', ascending=False)
                    importance_weight = model.get_booster().get_score(importance_type='weight')
                    importance_weight_df = pd.DataFrame({
                        'Feature': [feature_map.get(f, f) for f in importance_weight.keys()],
                        'Weight': importance_weight.values()
                    }).sort_values('Weight', ascending=False)
                    importance_cover = model.get_booster().get_score(importance_type='cover')
                    importance_cover_df = pd.DataFrame({
                        'Feature': [feature_map.get(f, f) for f in importance_cover.keys()],
                        'Cover': importance_cover.values()
                    }).sort_values('Cover', ascending=False)
                    import shap
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X_train)
                    shap_importance = np.abs(shap_values.values).mean(axis=0)
                    shap_importance_df = pd.DataFrame({
                        'Feature': df.drop(columns=target_column).columns,
                        'SHAP_Importance': shap_importance
                    }).sort_values('SHAP_Importance', ascending=False)
                    def normalize(df, column):
                        df[column + '_norm'] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                        return df
                    importance_gain_df = normalize(importance_gain_df, 'Gain')
                    importance_weight_df = normalize(importance_weight_df, 'Weight')
                    importance_cover_df = normalize(importance_cover_df, 'Cover')
                    feature_importance = normalize(feature_importance, 'Importance')
                    shap_importance_df = normalize(shap_importance_df, 'SHAP_Importance') 
                    combined_df = (
                        feature_importance[['Feature', 'Importance_norm']]
                        .merge(importance_gain_df[['Feature', 'Gain_norm']], on='Feature')
                        .merge(importance_weight_df[['Feature', 'Weight_norm']], on='Feature')
                        .merge(importance_cover_df[['Feature', 'Cover_norm']], on='Feature')
                        .merge(shap_importance_df[['Feature', 'SHAP_Importance_norm']], on='Feature')
                    )
                    weights = {
                        'Importance_norm': 0.2,
                        'Gain_norm': 0.3,
                        'Weight_norm': 0.1,
                        'Cover_norm': 0.1,
                        'SHAP_norm': 0.3
                    }
                    combined_df['Composite_Score'] = (
                        combined_df['Importance_norm'] * weights['Importance_norm'] +
                        combined_df['Gain_norm'] * weights['Gain_norm'] +
                        combined_df['Weight_norm'] * weights['Weight_norm'] +
                        combined_df['Cover_norm'] * weights['Cover_norm'] +
                        combined_df['SHAP_Importance_norm'] * weights['SHAP_norm']
                    )
                    combined_df = combined_df.sort_values('Composite_Score', ascending=False)
                    os.makedirs(os.path.dirname(save_importance_path), exist_ok=True)
                    combined_df.to_csv(save_importance_path, index=False)
                else:
                    combined_df = pd.read_csv(save_importance_path)
                topk=cfg.get("top_k_features", 100)
                if cfg.get("retrain_with_most_important_features", False) and cfg.model.type == 'xgboost':
                    retrain_df = df[combined_df[:topk]['Feature'].tolist() + [target_column]]
                    X_train, X_test, y_train, y_test = split_dataset(retrain_df, cfg.test_split_ratio, target_column)
                    logging.info(f"使用与{target_column}最相关的前{topk}个特征进行训练")
                    model = trainer.train(X_train, y_train, random_state=seed)
                if cfg.get("use_fields", None):
                    use_fields = cfg.get("use_fields", [])
                    if not isinstance(use_fields, list):
                        use_fields = [use_fields]
                    retrain_df = df[use_fields + [target_column]]
                    X_train, X_test, y_train, y_test = split_dataset(retrain_df, cfg.test_split_ratio, target_column)
                    logging.info(f"使用包含字段{use_fields}的特征进行训练")
                    model = trainer.train(X_train, y_train, random_state=seed)
                logging.info(f"模型已保存到 {save_model_path}")
                predictor = Predictor(save_model_path)
                y_pred = predictor.predict(X_test)
                scores.append(calculate_pps(y_test, y_pred))
                r2_scores.append(r2_score(y_test, y_pred))
            print(f"Product: {product_name}, Target: {target_column}, Mean PPS: {np.mean(scores)}, Mean R2: {np.mean(r2_scores)}")
            results[product_name][target_column] = np.mean(scores)
            results[product_name][f"{target_column}_R2"] = np.mean(r2_scores)

    # 结果汇总与平均
    logging.info("\n--- 评测结果 ---")
    # 先对每个seed的results做overall
    for target_column in cfg.data_loader.target_columns:
        total_score = 0
        total_score_r2 = 0
        for product_name, product_score in results.items():
            total_score += product_score[target_column]
            total_score_r2 += product_score.get(f"{target_column}_R2", 0)
        results['overall'][target_column] = total_score / 3
        results['r2'][target_column] = total_score_r2 / 3
    # 对所有seed的所有字段结果做平均
    final_score = np.mean(list(results['overall'].values()))
    # 打印所有seed的结果和平均结果
    print_pretty_results(results, final_score)
    write_score_to_csv(results, final_score, os.path.join(cfg.model_save_dir, cfg.experiment_name))

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