from typing import Dict, Callable

FREQUENCY_MAP_PROFIT = {
    '原油：常减压：独立炼厂：装置毛利：中国（周）': 'W', 
    '原油：炼油工序：毛利：中国：主营炼厂（周）': 'W', 
    '原油：炼油工序：毛利：山东：独立炼厂（周）': 'W', 
    '柴油：裂解：价差：中国：主营销售公司（日）': 'D', 
    '汽油：裂解：价差：中国：主营销售公司（日）': 'D'
}
# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_PROFIT: Dict[str, Callable] = {

}