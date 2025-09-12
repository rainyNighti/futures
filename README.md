# 比赛信息
中国石化第一节人工智能创新大赛

赛题7： 国内能源化工品种期货价格预测

[比赛赛题链接](https://aicup.sinopec.com/competition/SINOPEC-07/)

# 环境
```
conda create -n futures python=3.10
conda activate futures
pip install -r requirements.txt
```

# 文件结构简介
```
futures/
├── analyze_data/               # 是对每个表格的探索性分析,代码都是一样的,只是一个csv一个路径
├── configs/
│   └── base_config.yaml
├── saved_models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py       # 将多个表格数据汇集为一张总表
│   │   └── dataset.py           # split + 划分X和y
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessing.py     # 特征工程管道
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py        
│   └── utils/
│       ├── __init__.py
│       └── config.py            # 用于加载配置
├── main.py                     
└── requirements.txt
```

# 运行
单卡就可以运行的很快
python run.py --config configs/base_config.yaml


# 配置参数动态覆盖（extra_params用法）

支持通过命令行 --extra_params 传递参数，动态覆盖配置文件内容，支持嵌套dict、list结构。

## 基本用法

- 多个参数用英文逗号分隔：
	```bash
	python run.py --config configs/base_config.yaml --extra_params key1=val1,key2=val2
	```

- 支持嵌套：
	```bash
	python run.py --config configs/base_config.yaml --extra_params outer.inner=value
	```

- 支持list索引（如pipeline第3步），如果没有则会添加list元素：
	```bash
	python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.2.name=add_close_spread
	```
	这会将配置文件中的 `preprocessing_pipeline[2]["name"]` 覆盖为 `add_close_spread`。

- 也支持同时修改多个list项：
	```bash
	python run.py --config configs/base_config.yaml --extra_params preprocessing_pipeline.0.name=fill_miss_value,preprocessing_pipeline.1.name=frequency_encode_non_numeric
	```

- 支持list类型参数：
	```bash
	python run.py --config configs/base_config.yaml --extra_params some_list_param=[1,2,3]
	```
	解析后会变为Python的list类型。

## 注意事项
- 嵌套路径用英文点号`.`分隔，list索引用数字。
- 复杂嵌套如 `a.0.b.1.c=val` 也支持。
- 只支持简单类型（int/float/bool/list/str），复杂对象建议在yaml中配置。

# yaml文件配置变量教程
- 定义变量
```yaml
# 1. 为单个值设置锚点
default_learning_rate: &lr 0.05

# 2. 为列表设置锚点
admin_users: &admins
  - alice
  - bob

# 3. 为一个完整的字典/对象设置锚点
default_db_config: &db_defaults
  host: localhost
  port: 5432
  user: admin
  pool_size: 10

# 4. 为一个嵌套的字典/对象设置锚点
base_preprocessing_pipeline: &base_preprocessing_pipeline
  trade_pipeline:
    - type: 'aggregate_major_contracts'
    - type: 'add_historical_features' # 函数名
      n_lags: 20  # 使用历史N帧
    - type: 'add_y_targets'
      target_column: '收盘价_主力合约'
      future_steps: [5, 10, 20]

  supply_pipeline:
    - n_lags_m: 12  # 使用历史12个月的数据
      n_lags_w: 20  # 使用历史20周
```

- 使用变量
```yaml
# 1. 使用单个值的别名
model_params:
  learning_rate: *lr  # 这里的值会被解析为 0.05
  optimizer: 'adam'

# 2. 使用列表的别名
permissions:
  project_x:
    read_write: *admins # 这里会被解析为 ['alice', 'bob']

# 3. 使用字典的别名
production_database:
  <<: *db_defaults    # 特殊用法：合并/继承下面的键值对
  host: prod.db.server.com # 覆盖从别名继承过来的 host
  user: prod_user          # 覆盖 user

development_database:
  <<: *db_defaults    # 同样继承默认配置
  # 这里没有覆盖，所以 host, port, user, pool_size 都和默认值一样

# 4. 使用嵌套字典别名
sc_T_5_pipeline: 
  <<: *base_preprocessing_pipeline
  supply_pipeline:
    # - n_lags_m: 12  注意！这一行会消失，需要完整覆盖嵌套字典中的所有内容
    - n_lags_w: 2000000  # 这一行会覆盖
      n_lags_hhhhhhhh: 0	# 这一行新增
```