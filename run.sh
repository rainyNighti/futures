#!/bin/bash

# base config
python run.py --config configs/base_config.yaml

# test different seeds
# python run.py --config configs/base_config.yaml --extra_params seed=43,model.params.seed=43,experiment_name=seed43
# python run.py --config configs/base_config.yaml --extra_params seed=44,model.params.seed=44,experiment_name=seed44
# python run.py --config configs/base_config.yaml --extra_params seed=45,model.params.seed=45,experiment_name=seed45
# python run.py --config configs/base_config.yaml --extra_params seed=46,model.params.seed=46,experiment_name=seed46
# python run.py --config configs/base_config.yaml --extra_params seed=47,model.params.seed=47,experiment_name=seed47
# python run.py --config configs/base_config.yaml --extra_params seed=48,model.params.seed=48,experiment_name=seed48
# python run.py --config configs/base_config.yaml --extra_params seed=49,model.params.seed=49,experiment_name=seed49
# python run.py --config configs/base_config.yaml --extra_params seed=50,model.params.seed=50,experiment_name=seed50
# python run.py --config configs/base_config.yaml --extra_params seed=51,model.params.seed=51,experiment_name=seed51
# python run.py --config configs/base_config.yaml --extra_params seed=52,model.params.seed=52,experiment_name=seed52

# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_close_spread,experiment_name=add_close_spread
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_settle_spread,experiment_name=add_settle_spread
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_highlow_spread_range,experiment_name=add_highlow_spread_range
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_open_gap_spread,experiment_name=add_open_gap_spread
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_volume_ratio,experiment_name=add_volume_ratio
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_volume_change,experiment_name=add_volume_change
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_oi_transfer,experiment_name=add_oi_transfer
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_oi_ratio,experiment_name=add_oi_ratio
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_intraday_volatility,experiment_name=add_intraday_volatility
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_relative_strength,experiment_name=add_relative_strength
# python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_volume_direction,experiment_name=add_volume_direction