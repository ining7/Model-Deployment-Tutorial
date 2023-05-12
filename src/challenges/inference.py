import onnxruntime
import numpy as np
import cv2
import interpolation_training
import conversion

input_factor = np.array([1, 1, 4, 4], dtype=np.float32)
ort_session = onnxruntime.InferenceSession("srcnn3.onnx")
ort_inputs = {'input': interpolation_training.input_img, 'factor': input_factor}
ort_output = ort_session.run(None, ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort_3.png", ort_output)