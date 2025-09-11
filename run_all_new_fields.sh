#!/bin/bash

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


features=(
  add_open_gap_spread
  add_highlow_spread_range
  add_intraday_volatility
)

# use the above features as a base
python run.py --config configs/base_config.yaml --extra_params experiment_name=base_with_some_features,preprocessing_pipeline.2.name=add_open_gap_spread,preprocessing_pipeline.3.name=add_highlow_spread_range,preprocessing_pipeline.4.name=add_intraday_volatility