import yaml
from box import Box

def convert_str_to_types(value: str):
    """ support int, float, bool, list """
    if value.isdigit():
        return int(value)
    try:
        float_value = float(value)
        if '.' in value:
            return float_value
        return int(float_value)
    except ValueError:
        pass
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.startswith('[') and value.endswith(']'):
        # 解析列表
        list_items = value[1:-1].split(',')
        str_value = [item.strip() for item in list_items]
        value = [convert_str_to_types(item) for item in str_value]
        types_of_list = set(type(item) for item in value)
        if len(types_of_list) == 1:
            return value
        else:
            return str_value # 如果列表中类型不一致，返回字符串列表
    return value

def add_param(config: Box, key_route: str, value):
    """递归地在Box对象中添加或更新参数，支持list索引。"""
    keys = key_route.split('.')
    d = config
    for idx, k in enumerate(keys[:-1]):
        if isinstance(d, list) and k.isdigit():
            k = int(k)
            while len(d) <= k:
                d.append({})
            d = d[k]
        else:
            if k not in d or not isinstance(d[k], (dict, list)):
                if idx+1 < len(keys)-1 and keys[idx+1].isdigit():
                    d[k] = []
                else:
                    d[k] = {}
            d = d[k]
    last_k = keys[-1]
    if isinstance(d, list) and last_k.isdigit():
        last_k = int(last_k)
        while len(d) <= last_k:
            d.append(None)
        d[last_k] = convert_str_to_types(value)
    else:
        d[last_k] = convert_str_to_types(value)

def load_config(file_path: str, extra_params: str = None) -> Box:
    """加载YAML配置文件并返回一个Box对象，方便访问。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    base_pipeline_config_name = "base_preprocessing_pipeline"
    other_pipeline_config_names = [
        "brent_T_5_pipeline",
        "brent_T_10_pipeline",
        "brent_T_20_pipeline",
        "sc_T_5_pipeline",
        "sc_T_10_pipeline",
        "sc_T_20_pipeline",
        "wti_T_5_pipeline",
        "wti_T_10_pipeline",
        "wti_T_20_pipeline"
    ]
    if extra_params:
        # 解析额外的配置参数，支持list索引
        extra_params = extra_params.split(',')
        base_pipeline_configs = [c for c in extra_params if c.startswith(base_pipeline_config_name)]
        non_base_configs = [c for c in extra_params if not c.startswith(base_pipeline_config_name)]
        for param in base_pipeline_configs:
            try:
                key, value = param.split('=')
                add_param(config_dict, key, value)
            except ValueError:
                print(f"Warning: Unable to parse extra param '{param}'. Expected format 'key=value', also make sure the key route exists.")
                continue
        for other_pipeline_config_name in other_pipeline_config_names:
            config_dict[other_pipeline_config_name] = config_dict.get(other_pipeline_config_name, {}).copy()
        for param in non_base_configs:
            try:
                key, value = param.split('=')
                if key.startswith(other_pipeline_config_name):
                    add_param(config_dict, key, value)
            except ValueError:
                print(f"Warning: Unable to parse extra param '{param}'. Expected format 'key=value', also make sure the key route exists.")
                continue
        
    return Box(config_dict)