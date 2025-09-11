# base config
python run.py --config configs/base_config.yaml

python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_close_spread,experiment_name=add_close_spread
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_settle_spread,experiment_name=add_settle_spread
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_highlow_spread_range,experiment_name=add_highlow_spread_range
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_open_gap_spread,experiment_name=add_open_gap_spread
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_volume_ratio,experiment_name=add_volume_ratio
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_volume_change,experiment_name=add_volume_change
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_volume_direction,experiment_name=add_volume_direction
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_oi_change,experiment_name=add_oi_change
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_oi_transfer,experiment_name=add_oi_transfer
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_oi_ratio,experiment_name=add_oi_ratio
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_intraday_volatility,experiment_name=add_intraday_volatility
python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_relative_strength,experiment_name=add_relative_strength