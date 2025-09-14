#!/bin/bash

# 参数空间
ae_hidden_list=(512 1024)
ae_code_list=(32 64)
mlp_hidden_list=(512 1024)
dropout_list=(0.2 0.3)
noise_std_list=(0.05 0.1)
learning_rate_list=(1e-4 3e-5)
recon_epochs=40
mlp_epochs=100

for ae_hidden in "${ae_hidden_list[@]}"; do
  for ae_code in "${ae_code_list[@]}"; do
    for mlp_hidden in "${mlp_hidden_list[@]}"; do
      for dropout in "${dropout_list[@]}"; do
        for noise_std in "${noise_std_list[@]}"; do
          for learning_rate in "${learning_rate_list[@]}"; do
            experiment_name="aemlp_ae${ae_hidden}_code${ae_code}_mlp${mlp_hidden}_drop${dropout}_noise${noise_std}_lr${learning_rate}"
            python train.py configs/base_config.yaml \
              --extra_params model.type=aemlp,model.ae_hidden=${ae_hidden},model.ae_code=${ae_code},model.mlp_hidden=${mlp_hidden},model.dropout=${dropout},model.noise_std=${noise_std},model.learning_rate=${learning_rate},model.recon_epochs=${recon_epochs},model.mlp_epochs=${mlp_epochs},experiment_name=${experiment_name}
          done
        done
      done
    done
  done
done

# final result: best model is hidden 512, code 32, mlp 512, dropout 0.3, noise any, lr 1e-4