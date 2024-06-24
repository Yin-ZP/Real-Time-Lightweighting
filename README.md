# 面向推理阶段的模型实时轻量化技术开源仿真项目

## 文件说明
- `train.py`：模型训练程序
- `test.py`：模型轻量化测试程序
- `resnet18_cut_net.py`：ResNet18模型结构程序
- `resnet18_cut_val.py`：ResNet18模型轻量化测试程序
- `mobilenetv1_cut_net.py`：Mobilenetv1模型结构程序
- `mobilenetv1_cut_val.py`：Mobilenetv1模型轻量化测试程序
- `cifra_get_parameter.py`：模型定点数量化处理程序
- `cifra_getcoe.py`：coe文件生成程序
- `madd_cpt.py`：模型推理降低计算量计算程序

## 使用说明
- 对程序中剪枝条件函数阈值进行修改，从而观察不同剪枝阈值下的模型实时轻量化效果，然后根据剪枝情况对模型整体计算量进行计算，得到剪枝阈值与模型下降精度与降低计算量之间的关系。
- mbv1模型为已经训练好的模型参数，可直接进行测试。

## 开源项目信息
- **单位**：西安电子科技大学SCL实验室
- **开源人员**：Yang Qinghai、Yin Zepeng
- **实验环境**：Windows 10 64bit、NVIDIA GeForce RTX 1060SUPER
- **仿真平台及所需工具**： PyCharm、 PyTorch

