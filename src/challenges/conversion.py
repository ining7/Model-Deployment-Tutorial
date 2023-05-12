import torch
import torch.onnx
import training
import interpolation_training

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(interpolation_training.model, (x, interpolation_training.factor),
                      "srcnn3.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])
