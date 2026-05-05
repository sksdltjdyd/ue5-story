import torch
from torchvision import models
import onnxruntime as ort

# ONNX 설치 확인
print(f"ONNX 버전 확인: {ort.__version__}")
print(f"사용 가능 가속기: {ort.get_available_providers()}")
print(f"엔진 사용 가능")

# ONNX 파일 메모리에 로드
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights = weights)
model.eval()

# 모델 구조 확인
print(model)