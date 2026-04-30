# Net 项目构建与使用说明

## 1. 项目简介

本项目是一个基于 PyTorch 的房价预测训练示例，使用 `HousePrice` 数据集完成以下流程：

- 读取训练集和测试集
- 进行数值缺失值填充与类别特征独热编码
- 划分训练集与验证集
- 构建三层全连接神经网络进行训练
- 输出测试集预测结果
- 保存训练过程和验证结果可视化图表

项目适合用于学习一个完整的机器学习训练流程，包括数据预处理、模型训练、评估、推理与结果导出。

## 2. 项目目录结构

```text
Net/
|-- code/
|   |-- main.py          # 程序入口
|   |-- config.py        # 训练参数与路径配置
|   |-- data.py          # 数据读取与预处理
|   |-- dataset.py       # Dataset 和 DataLoader 构建
|   |-- model.py         # 模型定义
|   |-- trainer.py       # 训练、验证、测试逻辑
|   |-- pipeline.py      # 训练历史记录与日志
|   |-- visualizer.py    # 训练曲线与验证结果可视化
|   `-- __init__.py
|-- data/
|   `-- HousePrice/
|       |-- train.csv
|       `-- test.csv
|-- output/
|   |-- test_target.csv  # 测试集预测结果
|   `-- plots/           # 训练曲线和验证可视化结果
`-- doc/
    `-- readme.md
```

## 3. 环境构建

### 3.1 Python 版本建议

建议使用 `Python 3.10+`。

### 3.2 创建虚拟环境

在 `Net` 目录下执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

如果 PowerShell 默认禁止脚本执行，可先运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 3.3 安装依赖

本项目代码中实际使用到的主要依赖如下：

- `torch`
- `pandas`
- `numpy`
- `matplotlib`

可直接安装：

```powershell
pip install torch pandas numpy matplotlib
```

安装完成后可用以下命令检查：

```powershell
python -c "import torch, pandas, numpy, matplotlib; print('env ok')"
```

## 4. 数据准备

项目默认读取以下文件：

- `Net/data/HousePrice/train.csv`
- `Net/data/HousePrice/test.csv`

其中：

- `train.csv` 需要包含目标列 `SalePrice`
- `test.csv` 需要包含 `Id` 列，用于生成最终提交结果

代码会自动完成以下预处理：

1. 从训练集提取 `SalePrice` 作为标签
2. 合并训练特征与测试特征，保证独热编码维度一致
3. 对数值特征缺失值使用均值填充
4. 对类别特征执行 `one-hot encoding`
5. 按配置比例划分训练集和验证集

## 5. 运行方式

### 5.1 启动训练

进入代码目录后运行：

```powershell
cd D:\Work\ML\Net\code
python main.py
```

程序执行后会完成：

- 模型训练
- 验证集评估
- 测试集推理
- 结果文件保存
- 图表输出

### 5.2 训练日志

训练过程中终端会按 epoch 输出类似日志：

```text
Epoch 1/10 | train_loss=..., train_rmse=... | val_loss=..., val_rmse=...
```

这些指标可用于判断模型是否正常收敛。

## 6. 输出结果说明

训练完成后，主要输出内容如下：

### 6.1 预测结果文件

文件路径：

```text
Net/output/test_target.csv
```

该文件包含两列：

- `Id`
- `SalePrice`

可用于提交预测结果或进一步分析。

### 6.2 可视化图表

图表默认保存在：

```text
Net/output/plots/
```

生成内容包括：

- `loss_curve.png`：训练集与验证集 Loss 曲线
- `rmse_curve.png`：训练集与验证集 RMSE 曲线
- `val_actual_vs_pred.png`：验证集真实值与预测值散点图
- `val_residual_hist.png`：验证集残差分布图
- `val_residual_scatter.png`：验证集残差散点图

## 7. 关键配置说明

主要配置位于 `Net/code/config.py`：

- `EPOCHS`：训练轮数
- `BATCH_SIZE`：批大小
- `LEARNING_RATE`：学习率
- `VALID_RATIO`：验证集比例
- `RANDOM_SEED`：随机种子
- `HIDDEN_DIM1`、`HIDDEN_DIM2`：隐藏层维度
- `TEST_TARGET_PATH`：测试输出文件路径
- `PLOTS_DIR`：图表输出目录

如果你想调整训练效果或运行速度，可以优先修改这些参数。

## 8. 模型结构说明

模型定义在 `Net/code/model.py`，当前是一个简单的多层感知机：

- 输入层：根据预处理后的特征维度自动确定
- 隐藏层 1：`Linear -> ReLU`
- 隐藏层 2：`Linear -> ReLU`
- 输出层：`Linear -> 1`

该结构适合作为表格数据回归任务的入门版本，后续可以继续扩展为更复杂的模型。

## 9. 常见使用流程

推荐按以下顺序使用本项目：

1. 准备好 `train.csv` 和 `test.csv`
2. 创建虚拟环境并安装依赖
3. 进入 `Net/code`
4. 运行 `python main.py`
5. 到 `Net/output` 查看预测结果和图表
6. 根据效果调整 `config.py` 中的超参数后重新训练

## 10. 常见问题

### 10.1 找不到数据文件

请确认以下路径存在：

```text
Net/data/HousePrice/train.csv
Net/data/HousePrice/test.csv
```

### 10.2 缺少第三方库

如果运行时报 `ModuleNotFoundError`，说明依赖未安装完整，请重新执行：

```powershell
pip install torch pandas numpy matplotlib
```

### 10.3 输出目录不存在

项目会自动创建输出目录，一般不需要手动新建。如果没有看到结果，请确认程序是否正常执行结束。

## 11. 后续可扩展方向

如果后续要继续完善项目，可以考虑增加：

- `requirements.txt`，统一依赖管理
- 模型权重保存与加载
- 更丰富的评价指标
- 命令行参数支持
- 更完善的实验记录与日志系统

