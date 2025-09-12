import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
try:
    import category_encoders as ce
except ImportError:
    print("Warning: 'category_encoders' library not found. Binary encoding will not be available.")
    print("Please install it using: pip install category_encoders")
    ce = None

def _get_columns_to_process(df: pd.DataFrame, include_columns: list = None, exclude_columns: list = None) -> list:
    """内部辅助函数，用于根据包含和排除规则确定要处理的列列表。"""
    if include_columns:
        columns_to_process = list(include_columns)
    else:
        # 自动检测所有非数值类型的列
        columns_to_process = df.select_dtypes(exclude=np.number).columns.tolist()
    if exclude_columns:
        # 从待处理列表中排除指定的列
        columns_to_process = [col for col in columns_to_process if col not in exclude_columns]
    return columns_to_process

def label_encode(df: pd.DataFrame, include_columns: list = None, exclude_columns: list = None) -> pd.DataFrame:
    """ 对指定的列进行标签编码。 """
    df_encoded = df.copy()
    columns_to_process = _get_columns_to_process(df, include_columns, exclude_columns)
    for col in columns_to_process:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
    return df_encoded

def one_hot_encode(df: pd.DataFrame, include_columns: list = None, exclude_columns: list = None, drop_first: bool = False) -> pd.DataFrame:
    """
    对指定的列进行独热编码。
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame。
    include_columns (list, optional): 需要编码的列。如果为 None，则自动选择所有非数值列。
    exclude_columns (list, optional): 从自动选择或 include_columns 中排除的列。
    drop_first (bool): 是否删除每个特征的第一个类别以避免多重共线性。
    
    返回:
    pd.DataFrame: 经过编码处理后的 DataFrame。
    """
    columns_to_process = _get_columns_to_process(df, include_columns, exclude_columns)

    if not columns_to_process:
        print("One-Hot Encoding: No columns to process.")
        return df.copy()

    print(f"One-Hot Encoding will be applied to: {columns_to_process}")
    df_encoded = pd.get_dummies(df, columns=columns_to_process, drop_first=drop_first, dtype=int)
    return df_encoded

def ordinal_encode(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """
    根据提供的映射对指定的列进行有序编码。
    注意：此函数需要明确的映射关系，不参与自动列检测。
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame。
    column_mappings (dict): 一个字典，键是列名，值是类别的映射字典。
                            例如: {'Size': {'S': 1, 'M': 2, 'L': 3}}
    
    返回:
    pd.DataFrame: 经过编码处理后的 DataFrame 副本。
    """
    df_encoded = df.copy()
    print(f"Ordinal Encoding will be applied based on provided mappings for columns: {list(column_mappings.keys())}")
    for col, mapping in column_mappings.items():
        df_encoded[col] = df_encoded[col].map(mapping)
    return df_encoded

def frequency_encode(df: pd.DataFrame, include_columns: list = None, exclude_columns: list = None) -> pd.DataFrame:
    """
    对指定的列进行频率编码。
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame。
    include_columns (list, optional): 需要编码的列。如果为 None，则自动选择所有非数值列。
    exclude_columns (list, optional): 从自动选择或 include_columns 中排除的列。
    
    返回:
    pd.DataFrame: 经过编码处理后的 DataFrame 副本。
    """
    df_encoded = df.copy()
    columns_to_process = _get_columns_to_process(df, include_columns, exclude_columns)

    if not columns_to_process:
        print("Frequency Encoding: No columns to process.")
        return df_encoded
        
    print(f"Frequency Encoding will be applied to: {columns_to_process}")
    for col in columns_to_process:
        counts = df_encoded[col].value_counts()
        df_encoded[col] = df_encoded[col].map(counts)
    return df_encoded

def binary_encode(df: pd.DataFrame, include_columns: list = None, exclude_columns: list = None) -> pd.DataFrame:
    """
    对指定的列进行二进制编码。
    需要 'category_encoders' 库。
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame。
    include_columns (list, optional): 需要编码的列。如果为 None，则自动选择所有非数值列。
    exclude_columns (list, optional): 从自动选择或 include_columns 中排除的列。
    
    返回:
    pd.DataFrame: 经过编码处理后的 DataFrame。
    """
    if ce is None:
        raise ImportError("Please install the 'category_encoders' library to use binary_encode.")
        
    df_encoded = df.copy()
    columns_to_process = _get_columns_to_process(df, include_columns, exclude_columns)
    
    if not columns_to_process:
        print("Binary Encoding: No columns to process.")
        return df_encoded
        
    print(f"Binary Encoding will be applied to: {columns_to_process}")
    encoder = ce.BinaryEncoder(cols=columns_to_process, return_df=True)
    df_encoded = encoder.fit_transform(df_encoded)
    return df_encoded

