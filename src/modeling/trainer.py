import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
import os
import logging

class ModelTrainer:
    def __init__(self, model_config: dict, model_dir: str):
        self.model_config = model_config
        self.model_dir = model_dir
        self.models = []
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_model(self):
        """根据配置初始化模型，未来可扩展支持不同类型的模型。"""
        if self.model_config.type == 'xgboost':
            return xgb.XGBRegressor(**self.model_config.params)
        elif self.model_config.type == 'catboost':
            return cb.CatBoostRegressor(**self.model_config.params)
        elif self.model_config.type == 'lightgbm':
            return lgb.LGBMRegressor(**self.model_config.params)
        elif self.model_config.type == 'adaboost':
            estimator = DecisionTreeRegressor(max_depth=self.model_config.params.get('max_depth', 8))
            return AdaBoostRegressor(
                estimator=estimator,
                n_estimators=self.model_config.params.get('n_estimators', 1000),
                learning_rate=self.model_config.params.get('learning_rate', 0.05),
                loss=self.model_config.params.get('loss', 'linear'),
                random_state=self.model_config.params.get('random_state', 42)
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_config.type}")

    def train(self, X_train, y_train, val_size: float = 0.1, random_state: int = 42):
        """
        为y的每一列训练一个模型，支持早停机制。
        """
        from sklearn.model_selection import train_test_split
        num_targets = y_train.shape[1]
        
        for i in range(num_targets):
            logging.info(f"正在训练目标 {i+1}/{num_targets} ...")
            y_target = y_train[:, i]
            # 划分训练集和验证集
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_target, test_size=val_size, random_state=random_state
            )
            model = self._get_model()
            # 训练时使用早停机制
            if self.model_config.type == 'lightgbm':
                model.fit(
                    X_tr, y_tr, eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.log_evaluation(-1)       # -1 → no logging, or use 100 for every 100 rounds
                    ]
                
                )
            elif self.model_config.type == 'adaboost':
                model.fit(
                    X_tr, y_tr,
                )
            else:
                model.fit(
                    X_tr, y_tr, eval_set=[(X_val, y_val)], 
                    verbose=False
                )

            model_path = os.path.join(self.model_dir, f'model_target_{i}.joblib')
            joblib.dump(model, model_path)
            logging.info(f"模型已保存至: {model_path}")
            self.models.append(model)
        return self.models