## SeansonLeeAI

SeasonLeeAI名称源自于我给李时珍老师取的英文名（doge），本项目是一个多模态中药鉴定与质量评估，准备基于图像（电子眼）和气味（电子鼻），实现对中药的鉴别与质量评估 

### 数据来源
来源于飞浆AI的公开数据集[中药材识别数据集](https://aistudio.baidu.com/datasetdetail/55190)


### 环境部署
 - 1 安装Anaconda
 - 2 创建虚拟环境

```cmd
conda create -n computervision python=3.7
```

- 3 在虚拟环境中安装TensorFlow

```cmd
conda activate computervision # 激活虚拟环境
pip install tensorflow==1.15.0 -i 
https://pypi.tuna.tsinghua.edu.cn/simple # 安装TensorFlow
```

- 4 在虚拟环境中安装torch,torchversion和torchsummary

然后进入两个文件按照路径下的cmd，通过清华镜像来安装(如果安装不上可以试试不用镜像)，这是可以查询[torch和torchversion对照版本的链接](https://blog.csdn.net/shiwanghualuo/article/details/122860521)
我的版本是torch:1.10.2，torchversion:0.11.3

```cmd
pip install torch-1.10.2+cpu-cp37-cp37m-win_amd64.whl # 安装torch
pip install torchvision-0.11.3+cpu-cp37-cp37m-win_amd64.whl # 安装torchversion
```

- 5 在pycharm中创建工程SeasonLeeAI


### 研发内容

本项目的研究内容分为五个部分，分别是数据预处理、中药图像分割模型的构建与改进、中药分类模型的微调与选择、


