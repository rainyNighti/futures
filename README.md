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