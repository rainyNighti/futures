#!/bin/bash

# base config
# python run.py --config configs/base_config.yaml

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

features=(
  add_close_spread
  add_settle_spread
  add_open_gap_spread
  add_highlow_spread_range
  add_oi_transfer
)

feature_count=${#features[@]}
comb_count=$((2**feature_count))

for ((i=0; i<comb_count; i++)); do
  params=""
  idx=2
  exp_name="exp"
  for ((j=0; j<feature_count; j++)); do
    if (( (i >> j) & 1 )); then
      params="${params}preprocessing_pipeline.${idx}.name=${features[j]},"
      exp_name="${exp_name}_${features[j]}"
      ((idx++))
    fi
  done
  params="${params,}"
  if [ -n "$params" ]; then
    output=$(python run.py --config configs/base_config.yaml --extra_params ${params}experiment_name=${exp_name} 2>&1)
  else
    output=$(python run.py --config configs/base_config.yaml --extra_params experiment_name=${exp_name}_none 2>&1)
  fi
  score=$(echo "$output" | grep -oE 'FinalAverageScore:[ ]*[0-9.]+')
  score_value=$(echo "$score" | grep -oE '[0-9.]+' | head -1)
  if [ -z "$score_value" ]; then
    score_value="NA"
  fi
  echo "${exp_name},${score_value}" >> result.csv
done


features=(
  add_oi_ratio
  add_intraday_volatility
  add_volume_ratio
  add_volume_change
  add_volume_direction
)

feature_count=${#features[@]}
comb_count=$((2**feature_count))

for ((i=0; i<comb_count; i++)); do
  params=""
  idx=2
  exp_name="exp"
  for ((j=0; j<feature_count; j++)); do
    if (( (i >> j) & 1 )); then
      params="${params}preprocessing_pipeline.${idx}.name=${features[j]},"
      exp_name="${exp_name}_${features[j]}"
      ((idx++))
    fi
  done
  params="${params,}"
  if [ -n "$params" ]; then
    output=$(python run.py --config configs/base_config.yaml --extra_params ${params}experiment_name=${exp_name} 2>&1)
  else
    output=$(python run.py --config configs/base_config.yaml --extra_params experiment_name=${exp_name}_none 2>&1)
  fi
  score=$(echo "$output" | grep -oE 'FinalAverageScore:[ ]*[0-9.]+')
  score_value=$(echo "$score" | grep -oE '[0-9.]+' | head -1)
  if [ -z "$score_value" ]; then
    score_value="NA"
  fi
  echo "${exp_name},${score_value}" >> result.csv
done