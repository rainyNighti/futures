import yaml
from box import Box

def load_config(file_path: str) -> Box:
    """加载YAML配置文件并返回一个Box对象，方便访问。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return Box(config_dict)