#!/bin/bash

# base config
# python run.py --config configs/base_config.yaml

# python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=base,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_close_spread,experiment_name=add_close_spread,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_settle_spread,experiment_name=add_settle_spread,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_highlow_spread_range,experiment_name=add_highlow_spread_range,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_open_gap_spread,experiment_name=add_open_gap_spread,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_volume_ratio,experiment_name=add_volume_ratio,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_volume_change,experiment_name=add_volume_change,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_oi_transfer,experiment_name=add_oi_transfer,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_oi_ratio,experiment_name=add_oi_ratio,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_intraday_volatility,experiment_name=add_intraday_volatility,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_relative_strength,experiment_name=add_relative_strength,num_seeds=10,force_reprocess=True
# python run.py --config configs/no_fundamental.yaml --extra_params base_preprocessing_pipeline.trade_pipeline.3.type=add_volume_direction,experiment_name=add_volume_direction,num_seeds=10,force_reprocess=True

# python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=base,num_seeds=10,use_fields.100=欧元区:HICP\(调和CPI\):当月同比_M_lag_1 --debug

python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：日均产量预估值：美国（周）_W_lag_18,num_seeds=10,use_fields.100=原油：日均产量预估值：美国（周）_W_lag_18
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：加工量：美国（周）_W_lag_1,num_seeds=10,use_fields.100=原油：加工量：美国（周）_W_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_OPEC：原油：产量：伊拉克（月）_M_lag_1,num_seeds=10,use_fields.100=OPEC：原油：产量：伊拉克（月）_M_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_EIA：原油和石油产品（不包括战略石油储备）：库存：美国（周）_W_lag_5,num_seeds=10,use_fields.100=EIA：原油和石油产品（不包括战略石油储备）：库存：美国（周）_W_lag_5
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_NYMEX:持仓数量:非商业空头持仓:轻质低硫原油\(WTI原油\)_W_lag_1,num_seeds=10,use_fields.100=NYMEX:持仓数量:非商业空头持仓:轻质低硫原油\(WTI原油\)_W_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：产能利用率：美国（周）_W_lag_2,num_seeds=10,use_fields.100=原油：产能利用率：美国（周）_W_lag_2
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_EIA：战略储备石油：库存：全球（周）_W_lag_5,num_seeds=10,use_fields.100=EIA：战略储备石油：库存：全球（周）_W_lag_5
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_EIA：原油和石油产品（不包括战略石油储备）：库存：美国（周）_W_lag_6,num_seeds=10,use_fields.100=EIA：原油和石油产品（不包括战略石油储备）：库存：美国（周）_W_lag_6
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_中国:CPI:当月同比_M_lag_9,num_seeds=10,use_fields.100=中国:CPI:当月同比_M_lag_9
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：加工量：中国：主营炼厂（周）_W_lag_19,num_seeds=10,use_fields.100=原油：加工量：中国：主营炼厂（周）_W_lag_19
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：加工量：日本（周）_W_lag_2,num_seeds=10,use_fields.100=原油：加工量：日本（周）_W_lag_2
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_EIA：原油和石油产品（不包括战略石油储备）：库存：美国（周）_W_lag_8,num_seeds=10,use_fields.100=EIA：原油和石油产品（不包括战略石油储备）：库存：美国（周）_W_lag_8
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_EIA：原油：旋转钻机：在运行数量：美国（月）_M_lag_11,num_seeds=10,use_fields.100=EIA：原油：旋转钻机：在运行数量：美国（月）_M_lag_11
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_美国:CPI:同比_M_lag_1,num_seeds=10,use_fields.100=美国:CPI:同比_M_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_EIA：战略储备石油：库存：全球（周）_W_lag_2,num_seeds=10,use_fields.100=EIA：战略储备石油：库存：全球（周）_W_lag_2
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_欧元区:HICP\(调和CPI\):当月同比_M_lag_1,num_seeds=10,use_fields.100=欧元区:HICP\(调和CPI\):当月同比_M_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_NYMEX:持仓数量:非商业多头持仓:轻质低硫原油\(WTI原油\)_W_lag_19,num_seeds=10,use_fields.100=NYMEX:持仓数量:非商业多头持仓:轻质低硫原油\(WTI原油\)_W_lag_19
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：加工量：中国：主营炼厂（周）_W_lag_9,num_seeds=10,use_fields.100=原油：加工量：中国：主营炼厂（周）_W_lag_9
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_美国:PPI:最终需求:同比:季调_M_lag_4,num_seeds=10,use_fields.100=美国:PPI:最终需求:同比:季调_M_lag_4
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：产能利用率：美国（周）_W_lag_1,num_seeds=10,use_fields.100=原油：产能利用率：美国（周）_W_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：商业库存：美国（周）_W_lag_1,num_seeds=10,use_fields.100=原油：商业库存：美国（周）_W_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：产量：中国（月）_M_lag_1,num_seeds=10,use_fields.100=原油：产量：中国（月）_M_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_NYMEX:持仓数量:非商业空头持仓:轻质低硫原油\(WTI原油\).1_W_lag_1,num_seeds=10,use_fields.100=NYMEX:持仓数量:非商业空头持仓:轻质低硫原油\(WTI原油\).1_W_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_中国:CPI:当月同比_M_lag_8,num_seeds=10,use_fields.100=中国:CPI:当月同比_M_lag_8
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_EIA：原油和石油产品（不包括战略石油储备）：库存：美国（周）_W_lag_9,num_seeds=10,use_fields.100=EIA：原油和石油产品（不包括战略石油储备）：库存：美国（周）_W_lag_9
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_EIA：沥青和道路石油：库存：全球（周）_W_lag_1,num_seeds=10,use_fields.100=EIA：沥青和道路石油：库存：全球（周）_W_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_价差区间,num_seeds=10,use_fields.100=价差区间
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：常减压：独立炼厂：装置毛利：中国（周）_W_lag_17,num_seeds=10,use_fields.100=原油：常减压：独立炼厂：装置毛利：中国（周）_W_lag_17
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_中国:M2:同比_M_lag_11,num_seeds=10,use_fields.100=中国:M2:同比_M_lag_11
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：需求量：全球（月）_M_lag_1,num_seeds=10,use_fields.100=原油：需求量：全球（月）_M_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_原油：表观消费量：中国（月）_M_lag_1,num_seeds=10,use_fields.100=原油：表观消费量：中国（月）_M_lag_1
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_OPEC：原油：产量：沙特阿拉伯（月）_M_lag_1,num_seeds=10,use_fields.100=OPEC：原油：产量：沙特阿拉伯（月）_M_lag_1


python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_中国:房地产开发投资完成额:累计同比_M_lag_2,num_seeds=10,use_fields.100=中国:房地产开发投资完成额:累计同比_M_lag_2
python run.py --config configs/no_fundamental.yaml --extra_params experiment_name=add_,num_seeds=10,use_fields.100=
