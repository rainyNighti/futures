
def calculate_and_save_feature_importance(model: Any, X_train: pd.DataFrame, feature_names: List[str], save_path: Path) -> pd.DataFrame:
    """
    计算多种 XGBoost 特征重要性，并结合成综合得分后保存。
    """
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