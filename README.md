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