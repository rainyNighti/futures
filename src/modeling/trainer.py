import xgboost as xgb
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
        else:
            raise ValueError(f"不支持的模型类型: {self.model_config.type}")

    def train(self, X_train, y_train):
        """
        为y的每一列训练一个模型。
        """
        num_targets = y_train.shape[1]
        
        for i in range(num_targets):
            logging.info(f"正在训练目标 {i+1}/{num_targets} ...")
            
            y_target = y_train[:, i]
            model = self._get_model()
            
            # TODO: 实现早停机制需要从X_train中划分验证集
            model.fit(X_train, y_target, verbose=False)
            
            model_path = os.path.join(self.model_dir, f'model_target_{i}.joblib')
            joblib.dump(model, model_path)
            logging.info(f"模型已保存至: {model_path}")
            
            self.models.append(model)
        
        return self.models