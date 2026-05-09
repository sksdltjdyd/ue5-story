import torch
from torchvision import models

# MobileNet 추출 테스트 이유 : ResNet50은 성능이 좋지만 파라미터가 약 2,500만개로 무겁다. MobileNet은 약 350만개로 7배 이상 가볍다. 언리얼 클라이언트에 AI를 포함시켜 배포야 할 때 독립성과 속도, 그리고 용량 최적화는 매우 중요

# 1.mobilenet v2 모델 로드 및 평가 모드 전환
# 시험모드로 스위치를 전환하여 가중치를 고정
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights = weights)
model.eval()

# 2.더미데이터 생성 -> 더미데이터 생성 이유는 미리 자료값이 어떻게 들어갈지 알려주기 위해서이다
# mobilenet도 224x224 사이즈의 RGB(3채널) 이미지를 요구
dummy_input = torch.rand(1, 3, 224, 224)

# 3.onnx 추출
onnx_file_path = "mobilenet_unreal.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    export_params=True,
    opset_version=14,
    input_names=['input_image'],
    output_names=['output_image']
)

print(f"{onnx_file_path} 파일 추출 성공")