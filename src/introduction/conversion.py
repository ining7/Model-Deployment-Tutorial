import torch
import torch.onnx
import training

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        training.model,
        x,
        "srcnn.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])

''' 
torch.onnx.export 是 PyTorch 自带的把模型转换成 ONNX 格式的函数
前三个参数必选（要转换的模型、模型的任意一组输入、导出的ONNX文件的文件名）
'''