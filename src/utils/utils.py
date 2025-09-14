import os
import random
import numpy as np
import logging
from typing import Dict


def print_pretty_results(results: Dict[str, Dict[str, float]], final_score: float):
    """
    以美观、对齐的表格形式打印性能评估结果。

    Args:
        results (Dict[str, Dict[str, float]]): 包含评测结果的嵌套字典。
                                                 结构: {product_name: {target_column: score}}
                                                 应包含一个特殊的 'overall' 键。
        final_score (float): 最终的综合平均分。
    """
    if 'overall' not in results:
        logging.error("错误：结果字典中未找到 'overall' 项。")
        return
    for k, v in results.items():  # <-- 使用 .items()
        for k1, v1 in v.items(): # <-- 这里也需要 .items()
            # 检查 v1 是否为字典并且包含 'pps' 键
            if isinstance(v1, dict) and 'pps' in v1:
                v[k1] = v1['pps'] # 用 'pps' 的值替换原来的字典

    # 1. 提取所有指标名称和产品名称（不包括 'overall'）
    target_columns = list(results['overall'].keys())
    product_names = sorted([p for p in results.keys() if p != 'overall'])

    # 2. 动态计算每列的最佳宽度以保证对齐
    # 产品列的宽度由最长的产品名或 "Overall 平均分" 或列标题 "产品名称" 决定
    product_col_width = max(
        [len(p) for p in product_names] + [len("Overall 平均分"), len("产品名称")]
    ) + 4  # 增加 4 个字符作为边距

    # 每个分数咧的宽度由指标名称或分数的长度（例如 "-0.123456"）决定
    score_col_widths = {
        col: max(len(col), 10) + 4 for col in target_columns # 10位数字宽度，4位边距
    }

    # 3. 构建并打印表头
    header_parts = [f"{'产品名称':<{product_col_width}}"]
    for col in target_columns:
        header_parts.append(f"{col:^{score_col_widths[col]}}") # 居中对齐
    header = "".join(header_parts)
    
    separator = "=" * len(header)

    logging.info("\n" + separator)
    logging.info(f"{'--- 性能评测结果 ---':^{len(header)}}")
    logging.info(separator)
    logging.info(header)
    logging.info("-" * len(header))

    # 4. 打印每个产品的分数
    for product in product_names:
        row_parts = [f"{product:<{product_col_width}}"]
        for col in target_columns:
            # 使用 .get(col, 0.0) 避免因缺失数据而报错
            score = results[product].get(col, 0.0)
            row_parts.append(f"{score:^{score_col_widths[col]}.6f}")
        logging.info("".join(row_parts))

    # 5. 打印 Overall 平均分
    logging.info("-" * len(header))
    overall_row_parts = [f"{'Overall 平均分':<{product_col_width}}"]
    for col in target_columns:
        score = results['overall'].get(col, 0.0)
        overall_row_parts.append(f"{score:^{score_col_widths[col]}.6f}")
    logging.info("".join(overall_row_parts))
    
    # 6. 打印最终总分
    logging.info(separator)
    final_score_text = f"最终综合得分 (Final Score): {final_score:.6f}"
    logging.info(f"{final_score_text:>{len(header)}}") # 右对齐
    logging.info(separator + "\n")

def write_score_to_csv(results: Dict[str, Dict[str, float]], final_score: float, save_dir: str):
    """
    将评测结果写入CSV文件，方便后续分析和记录。

    Args:
        results (Dict[str, Dict[str, float]]): 包含评测结果的嵌套字典。
                                                 结构: {product_name: {target_column: score}}
                                                 应包含一个特殊的 'overall' 键。
        final_score (float): 最终的综合平均分。
        save_dir (str): 用于保存CSV文件的目录路径。
    """
    import csv
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, "evaluation_results.csv")

    if 'overall' not in results:
        logging.error("错误：结果字典中未找到 'overall' 项。")
        return

    target_columns = list(results['overall'].keys())
    product_names = sorted([p for p in results.keys() if p != 'overall'])

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        header = ['产品名称'] + target_columns + ['average']
        writer.writerow(header)

        # 写入每个产品的分数
        for product in product_names:
            row = [product] + [f"{results[product].get(col, 0.0):.6f}" for col in target_columns]
            row = row + [f"{np.mean([float(x) for x in row[1:]]):.6f}"]  # 计算该行的平均分
            writer.writerow(row)

        # 写入 Overall 平均分
        overall_row = ['Overall 平均分'] + [f"{results['overall'].get(col, 0.0):.6f}" for col in target_columns]
        overall_row = overall_row + [f"{final_score:.6f}"]  # 最后一列是 final_score
        writer.writerow(overall_row)

    logging.info(f"评测结果已保存至: {csv_file_path}")

def set_random_seed(seed=32):
    """
    设置所有相关库的随机数种子，以确保实验的可复现性。

    Args:
        seed (int): 您想要设置的种子数值。
    """
    print(f"setting seed to: {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 尝试为 TensorFlow 设置种子 (如果已安装)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # 对于 TensorFlow 1.x
        # tf.set_random_seed(seed)
    except ImportError:
        print("tensorflow not installed, skipping setting its seed")
        pass
        
    # 尝试为 PyTorch 设置种子 (如果已安装)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
            # 确保CUDA操作的确定性，可能会影响性能
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        print("pytorch not installed, skipping setting its seed")
        pass

    print(f"所有相关的随机数种子已设置为: {seed}")