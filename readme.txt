# 多光谱 + 文本特征融合产量预测模型 (ResNet18)

本项目基于 ResNet18 架构，融合多光谱图像特征（RGB、蓝、绿、红、红边、近红外、3T）与文本特征（18 维），用于作物产量（归一化产量）的回归预测。

## 目录结构
.
├── train.py # 主训练脚本
├── model_mul.py # ResNet18 多模态模型定义
├── Dataloder_mul.py # 自定义数据集加载类
├── Dataset/ # 数据集目录
│ ├── 1_20_ndyield_train.json # 训练集标签
│ ├── 1_20_ndyield_val.json # 验证集标签
│ └── 1_20_nd-yield.csv # 产量数据
└── 1_20_ValR18HTF/ # 模型保存目录（自动创建）

text

## 环境依赖

- Python 3.8+
- PyTorch 1.10+
- openpyxl
- scikit-learn
- numpy

安装命令：
```bash
pip install torch openpyxl scikit-learn numpy
数据准备
输入数据
多光谱图像：七个通道分别位于以下路径：

F:/wq-data/final/1-20/rgb/

F:/wq-data/final/1-20/blue/

F:/wq-data/final/1-20/green/

F:/wq-data/final/1-20/red/

F:/wq-data/final/1-20/rededge/

F:/wq-data/final/1-20/nir/

F:/wq-data/final/1-20/3T/

标签文件（JSON 格式）：

训练集：Dataset/1_20_ndyield_train.json

验证集：Dataset/1_20_ndyield_val.json

产量 CSV 文件：Dataset/1_20_nd-yield.csv

数据加载说明
MyDataset 类会同时加载：

多光谱图像（7 个通道）

文本特征（18 维，如环境或管理参数）

对应的归一化产量标签

模型配置
基模型：ResNet18（预训练权重可选）

输出维度：1（回归值）

文本特征维度：18

融合方式：特征拼接后接入全连接层

超参数设置
参数	值
训练批次大小	16
验证批次大小	8
学习率	0.0001
优化器	Adam
损失函数	MSELoss
训练轮数	500
设备	GPU (cuda:0) / CPU
运行训练
bash
python train.py
输出文件
训练结束后会生成两个 Excel 文件：

1. 1_20_ValR18HTF_metrics.xlsx
包含两个工作表：

Metrics：每个 epoch 的训练/验证指标

Epoch

Train Loss, Train R², Train MAE, Train RMSE

Val Loss, Val R², Val MAE, Val RMSE

Predictions：最佳验证集 R² 对应的预测值与真实值，以及最佳 R² 值

2. 1_20_ValR18HTF_para.xlsx
Training Parameters：记录训练参数与时间

训练/验证批次大小

模型参数量

每个 epoch 的训练时间

收敛状态

模型保存
每当验证集损失（Val Loss）下降时，保存模型权重至：

text
1_20_ValR18HTF/{epoch}_ResNet.pth
评估指标
训练过程实时计算并记录以下指标：

Loss：均方误差（MSE）

R² 决定系数：回归拟合优度

MAE：平均绝对误差

RMSE：均方根误差

注意事项
请确保所有数据路径存在且有读取权限。

若使用 GPU，请确认 CUDA 环境已配置。

数据加载中的 num_workers 可根据本地 CPU 核心数调整。

模型定义文件 model_mul.py 中 resnet18 需接收 text_feature_dim=18 参数。

训练耗时较长（500 轮），建议使用 GPU 加速。

常见问题
Q: 运行时提示 ModuleNotFoundError: No module named 'Dataloder_mul'
A: 请确保 Dataloder_mul.py 与训练脚本在同一目录，或将其添加到 Python 路径。

Q: 图像路径包含中文导致错误
A: 建议将数据放在全英文路径下，或设置合适的编码格式。

Q: 验证 R² 为负数
A: 可能模型欠拟合，可尝试增加训练轮数、调整学习率或增加模型复杂度。

作者
可根据需要补充。

许可证
MIT License（示例，请根据实际情况修改）