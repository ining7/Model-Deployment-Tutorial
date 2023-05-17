## ENV

```
python
pytorch
torchvision
onnxruntime
onnx
opencv-python
```

## Tutorial

### [Introduction](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/01_introduction_to_model_deployment.md)

  通过Pytorch训练超分辨率模型，将其转为ONNX描述的模型，在ONNX Runtime上运行。

```shell
cd ./src/introduction
python inference.py
```

- `training.py`：使用Pytorch训练超分辨率模型（如果脚本正常运行，一幅超分辨率的人脸照片会保存在“face_torch.png”中）
- `conversion.py`：转ONNX模型（如果脚本正常运行，目录下会新增一个"srcnn.onnx"的 ONNX 模型文件）
- `check_onnx.py`：检查模型格式是否正确（如果模型正确，打印"Model correct"）
- `inference.py`：使用ONNX Runtime运行模型（如果脚本正确运行，目录下会新增一个与“face_torch.png"内容一样的"face_ort.png"图片）

### [Challenges](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/02_challenges.md)

​	将上一节中的超分辨率模型修改成动态输入，通过自定义算子解决过程中出现的问题兼容性问题。

```shell
cd ./src/challenges
python inference.py
```

- `training.py`：如果脚本正常运行，目录下会新增一个与“face_torch.png"内容一样的"face_torch_2.png"图片
- `interpolation_training.py`：定义了一个PyTorch插值算子，并在模型中使用（如果脚本正常运行，目录下会新增一个与“face_torch.png"内容一样的"face_torch_3.png"图片）
- `conversion.py`：转ONNX模型
- `inference.py`：使用ONNX Runtime运行模型（如果脚本正确运行，目录下会新增一个与“face_torch.png"内容一样的"face_torch_3.png"图片）

### [Pytorch2onnx](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/03_pytorch2onnx.md)

​	ONNX作为中间表示，熟悉ONNX的技术细节能够规避和解决大量部署中出现的问题，本节主要介绍PyTorch转ONNX的细节。

#### PyTorch转ONNX

```shell
cd ./src/pytorch2onnx
```

##### 计算图导出方法 

1. `trace`跟踪（default）：只能通过实际运行一次模型的方法导出模型的静态图，无法识别出模型中的控制流
2. `script`脚本化：通过解析模型记录所有的控制流

​	通常在部署模型时不需要显式的把PyTorch模型转成TorchScript模型。

- `compare_conversion.py`：如果脚本正常运行，目录下会新增4个ONNX模型"model_2_script.onnx"，"model_3_script.onnx"，"model_2_trace.onnx"，"model_3_trace.onnx"，可以将他们使用Netron进行可视化。

##### 动态维数

- `dynamic_axes.py`：如果脚本正常运行，目录下会新增3个ONNX模型"model_static.onnx"，"model_dynamic_0.onnx"，"model_dynamic_23.onnx"，并且有以下输出

  ```shell
  Input[0] on model model_static.onnx succeed.
  Input[1] on model model_static.onnx error.
  Input[2] on model model_static.onnx error.
  Input[0] on model model_dynamic_0.onnx succeed.
  Input[1] on model model_dynamic_0.onnx succeed.
  Input[2] on model model_dynamic_0.onnx error.
  Input[0] on model model_dynamic_23.onnx succeed.
  Input[1] on model model_dynamic_23.onnx error.
  Input[2] on model model_dynamic_23.onnx succeed.
  [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: in for the following indices
   index: 0 Got: 2 Expected: 1
   Please fix either the inputs or the model.
  ```

##### 使用技巧

- 利用如`torch.onnx.is_in_onnx_export()`的函数对执行状态进行判断，使得推理时能执行不同的逻辑
- 利用中断判断tensor跟踪

#### PyTorch算子在ONNX中的兼容问题

​	在模型转换的过程中，PyTorch会跟踪执行前向推理，将算子翻译成ONNX中定义的算子并整合成计算图，在这个过程中可能会出现：

- 该算子可以一对一地翻译成一个 ONNX 算子；
- 该算子在 ONNX 中没有直接对应的算子，会翻译成一至多个 ONNX 算子；
- 该算子没有定义翻译成 ONNX 的规则，报错。

​	在**出现报错**时，我们应该考虑PyTorch算子与ONNX算子的对应情况，由于PyTorch算子是向ONNX对其的，所以应该先确认ONNX算子的定义情况，然后查看PyTorch定义的算子映射关系：

1. [ONNX算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md) - 算子变更表
2. [PyTorch中与ONNX有关的定义](https://github.com/pytorch/pytorch/tree/main/torch/onnx) - torch.onnx目录（阅读源码）

### [OnnxCustomOp](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/04_onnx_custom_op.md)

​	如何在PyTorch中支持更多的ONNX算子。

```shell
cd ./src/onnxOp
```

​	在PyTorch中支持更多的ONNX算子有以下角度：

- 实现PyTorch算子
  - 组合现有算子
  - 添加TorchScript算子
  - 添加普通C++拓展算子
- 增添映射关系
  - 为ATen算子添加符号函数
  - 为TorchScript算子添加符号函数
  - 封装成torch.autograd.Function并添加符号函数
- ONNX算子
  - 使用现有ONNX算子
  - 定义新ONNX算子

#### 补充ATen的描述映射规则

- 获取ATen中算子接口定义
- 添加符号函数
- 测试算子

```shell
cd ATen
python inference.py
```

- `asinhOp.py`：如果运行正确，那么会生成`asinh.onnx`模型
- `inference.py`：如果运行正确，不会出现出现报错和输出

#### 新增TorchScript算子，为现有TorchScript添加ONNX支持

以Deformable Convolution为例介绍[为现有TorchScript添加ONNX支持](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)的方法

- 获取原算子的前向推理接口
- 获取目标ONNX算子的定义
- 编写符号函数并绑定

```shell
cd ../TorchScript
python defineOp.py
```

- `defineOp.py`：如果运行正确，那么会生成`dcn.onnx`模型

#### 使用torch.autograd.Function

​	封装算子的C++调用接口

```shell
cd ../Wrapper
python wrapper.py
```

- `wrapper.py`：如果运行正确，那么会生成`my_add.onnx`模型

### [OnnxModelEditing](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/05_onnx_model_editing.md)

​	ONNX的底层实现、读取、子模型提取、调试。



### [IntroductionTensorrt](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/06_introduction_to_tensorrt.md)





### [TensorrtPlugin](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/07_write_a_plugin.md)



