import xgboost as xgb
import joblib
import os
import logging
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, model_config: dict, save_model_path: str):
        self.model_config = model_config
        self.save_model_path = save_model_path
        os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)

    def _get_model(self):
        if self.model_config.type == 'xgboost':
            return xgb.XGBRegressor(**self.model_config.params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_config.type}")

    def train(self, X_train, y_train, val_size: float = 0.1, random_state: int = 42):
        """
        为y的每一列训练一个模型，支持早停机制。
        """
        # 划分训练集和验证集
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state
        )
        model = self._get_model()
        # 训练时使用早停机制
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        joblib.dump(model, self.save_model_path)
        logging.info(f"模型已保存至: {self.save_model_path}")
        return model