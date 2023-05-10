import onnxruntime
import check_onnx
import training
import numpy as np
import cv2

ort_session = onnxruntime.InferenceSession("srcnn.onnx")
ort_inputs = {'input': training.input_img}
ort_output = ort_session.run(['output'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort.png", ort_output)