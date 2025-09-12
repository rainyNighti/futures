from typing import Dict, Callable

FREQUENCY_MAP_SUPPLY = {
    'OPEC：原油：产量：石油输出国组织+（月）': 'M', 
    'OPEC：原油：产量：沙特阿拉伯（月）': 'M', 
    'OPEC：原油：产量：科威特（月）': 'M', 
    'OPEC：原油：产量：尼日利亚（月）': 'M', 
    'OPEC：原油：产量：委内瑞拉（月）': 'M', 
    'OPEC：原油：产量：伊拉克（月）': 'M', 
    'OPEC：原油：产量：伊朗（月）': 'M', 
    'OPEC：战略储备石油：储备量：美国（月）': 
    'M', 'OPEC：原油：产量：阿联酋（月）': 'M', 
    '原油：日均产量预估值：美国（周）': 'W', 
    'EIA：原油：旋转钻机：在运行数量：美国（月）': 
    'M', '原油：产量：中国（月）': 'M'
}

# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_SUPPLY: Dict[str, Callable] = {

}