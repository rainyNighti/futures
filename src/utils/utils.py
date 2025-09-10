import os
import random
import numpy as np


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