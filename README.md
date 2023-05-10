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

