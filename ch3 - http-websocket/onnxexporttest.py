import torch
import torchvision.models as models

# 1. 모델 준비: 이미지 분류에 특화된 가벼운 사전 학습 모델(MobileNet V2) 로드
print("모델 다운로드 중...")
model = models.mobilenet_v2(pretrained=True)

# 2. 시험 모드로 전환: 학습용 스위치를 끄고 가중치를 고정(Bake)함 
model.eval() # [cite: 84]

# 3. 가짜 데이터(Dummy Input) 생성 [cite: 85]
# [배치 크기 1, 컬러 채널 3(RGB), 가로 224, 세로 224] 크기의 빈 박스 [cite: 86]
dummy_input = torch.rand(1, 3, 224, 224) # [cite: 86]

# 4. 언리얼 NNE 규격에 맞춰 ONNX 수출 (Export) [cite: 77]
print("언리얼용 ONNX 파일 굽는 중...")
torch.onnx.export( # [cite: 88]
    model,                      # 변환할 원본 모델 [cite: 22, 90]
    dummy_input,                # 가짜 데이터로 데이터 경로(Tracing) 생성 [cite: 23, 25, 91]
    "mobilenet_v2.onnx",        # 저장될 파일 이름 [cite: 24, 92]
    opset_version=14,           # FBX 버전처럼 언리얼 호환성(NNE)을 맞춤 
    input_names=['input_img'],  # 언리얼 C++에서 접근할 소켓(입력핀) 이름 [cite: 28, 29]
    output_names=['output_score'] # 결과값을 받아올 출구 이름 [cite: 30]
)

print("완료! 'mobilenet_v2.onnx' 파일이 생성되었습니다. 언리얼로 드래그 앤 드롭하세요!")