# 关系抽取-V2
本项目基于苏神的【GP-三元组抽取】做了进一步的改动， 将subject-object矩阵细化，同时解决了实体识别和关系抽取的问题


## 环境依赖
```python
conda create -n yourname python==3.10.0
conda activate yourname / source activate yourname
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install transformers==4.29.2
pip install rich==12.5.1
pip install flask
pip install gevent
```

## 项目结构
```html
|-- config.py
|-- README.md
|-- server.py
|-- train.py
|-- data
|   |-- chenyang
|       |-- dev_data.pkl
|       |-- fun_map.json
|       |-- entity_labels.json
|       |-- relation_labels.json
|       |-- preprocess.py
|       |-- train_data.pkl
|       |-- 样本.json
|-- utils
    |-- adversarial_training.py
    |-- base_model.py
    |-- functions.py
    |-- models.py
    |-- train_models.py
    |-- tokenizers.py
```

## 快速开始

```html
一、训练
1.【data/chenyang】位于【技术中心\0AI\DeepLearning\通用语料】，请自行拷贝
2.【样本.json】格式为从doccano平台导出关系抽取的数据集，其余平台可自行模仿
3.执行preprocess.py，获得3个json文件
4.修改config内data_dir参数
5.修改train.py脚本，新增一个if分支，然后执行train.py
二、服务
1.修改server.py内的model_name变量
2.运行server.py即可
3.可执行test.py验证
三、升级
参考docker升级手册和Dockerfile即可，可在107.46上部署升级
```

## 注意事项

```html
1.与GP一样，支持解决嵌套问题
2.样本量在2000左右时，会生成5个G级别的train_data.pkl，可用稀疏版交叉熵解决，计划下一版本优化
3.建议核查执行preprocess.py获得的3个json文件，尤其是【fun_map.json】，指示了关系与实体的对应关系
4.原则上讲，同一个关系的主体，客体类型都是唯一的，对于不唯一的情况，计划后续版本优化
```

 # 更新日志
 - 2023-11-21：修复了functions.py下get_relation_result_confi函数的bug
