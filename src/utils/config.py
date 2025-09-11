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


def load_config(file_path: str, extra_params: str = None) -> Box:
    """加载YAML配置文件并返回一个Box对象，方便访问。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    if extra_params:
        # 解析额外的配置参数，支持list索引
        for param in extra_params.split(','):
            try:
                key, value = param.split('=')
                key_route = key.split('.')
                d = config_dict
                for idx, k in enumerate(key_route[:-1]):
                    # 如果k是数字且d为list，则索引list
                    if isinstance(d, list) and k.isdigit():
                        k = int(k)
                        # 自动扩展list长度
                        while len(d) <= k:
                            d.append({})
                        d = d[k]
                    else:
                        # dict情况
                        if k not in d or not isinstance(d[k], (dict, list)):
                            # 判断下一个key是否为数字，决定初始化为list还是dict
                            if idx+1 < len(key_route)-1 and key_route[idx+1].isdigit():
                                d[k] = []
                            else:
                                d[k] = {}
                        d = d[k]
                # 处理最后一级
                last_k = key_route[-1]
                if isinstance(d, list) and last_k.isdigit():
                    last_k = int(last_k)
                    while len(d) <= last_k:
                        d.append(None)
                    d[last_k] = convert_str_to_types(value)
                else:
                    d[last_k] = convert_str_to_types(value)
            except ValueError:
                print(f"Warning: Unable to parse extra param '{param}'. Expected format 'key=value', also make sure the key route exists.")
                continue
    return Box(config_dict)