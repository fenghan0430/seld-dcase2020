# sound-localization

### 依赖

`tensorflow 2.15.1`

### 主要文件说明

`batch_feature_extraction.py` - 对数据集中的声音和标签进行预处理，数据集的路径需要在`parameter.py`中设置

`parameter.py` - 旧项目中最主要的配置文件，在新代码中已经被弃用

`sled.py` - 旧项目的训练代码，该训练代码包含和评估代码在内

`data.ipynb` - 对预处理过的项目再进行一次处理，使得数据可以直接加载并开始训练。将最后可以训练的数据保存为两个`.npy`文件

`keras_model.py` - 模型结构代码

`train.py` - 训练代码，读取数据集并训练

### ToDo

- [x] 编写评估代码，评估模型是否达到标准
- [ ] 重新训练模型，力求达到标准
