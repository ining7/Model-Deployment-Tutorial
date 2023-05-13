import torch

class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = torch.tensor(n, dtype=torch.int64)  # 将 n 转换为 torch.Tensor 类型
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        for i in torch.arange(self.n):  # 使用 torch.arange 替换 range
            x = self.conv(x)
        return x

models = [Model(2), Model(3)]
model_names = ['model_2', 'model_3']

for model, model_name in zip(models, model_names):
    dummy_input = torch.rand(1, 3, 10, 10)
    dummy_output = model(dummy_input)
    model_trace = torch.jit.trace(model, dummy_input)
    model_script = torch.jit.script(model)

    # 跟踪法与直接 torch.onnx.export(model, ...)等价
    torch.onnx.export(model_trace, dummy_input, f'{model_name}_trace.onnx', example_outputs=dummy_output)
    # 脚本化必须先调用 torch.jit.sciprt
    torch.onnx.export(model_script, dummy_input, f'{model_name}_script.onnx', example_outputs=dummy_output)
