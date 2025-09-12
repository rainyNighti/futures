from typing import Dict, Callable

FREQUENCY_MAP_FUND_HOLDING = {
    'NYMEX:持仓数量:非商业多头持仓:轻质低硫原油(WTI原油)': 'W', 
    'NYMEX:持仓数量:非商业空头持仓:轻质低硫原油(WTI原油)': 'W', 
    'NYMEX:持仓数量:商业多头持仓:轻质低硫原油(WTI原油)': 'W', 
    'NYMEX:持仓数量:非商业空头持仓:轻质低硫原油(WTI原油).1': 'W', 
    'ICE:IPE:期货持仓数量:管理基金多头持仓:原油': 'W', 
    'ICE:IPE:期货持仓数量:管理基金空头持仓:原油': 'W', 
    'ICE:IPE:期货持仓数量:生产商/贸易商/加工商/用户多头持仓:原油': 'W', 
    'ICE:IPE:期货持仓数量:生产商/贸易商/加工商/用户空头持仓:原油': 'W'
}
# 注册所有可用的预处理函数
PREPROCESSING_FUNCTIONS_FUND_HOLDING: Dict[str, Callable] = {

}