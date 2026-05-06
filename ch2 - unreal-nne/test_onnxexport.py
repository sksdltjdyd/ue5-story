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

# 가짜 데이터(dummy input) 만들기
# 언리얼에 패턴을 알려주는 쓰레기값  데이터
dummy_input = torch.rand(1,3,224,224)

# onnx 파일로 추출(언리얼 프로젝트 폴더에 넣을 파일)
onnx_file_path ="resnet50_unreal.onnx"
torch.onnx.export(
    model,                          # 1. 굽고 싶은 모델 객체
    dummy_input,                    # 2. 길을 뚫어줄 가짜 데이터
    "my_model.onnx",                # 3. 저장될 파일명
    export_params=True,             # (가중치/뇌 포함 여부. 항상 True)
    opset_version=14,               # 4. 언리얼 NNE가 잘 읽을 수 있는 FBX 버전
    input_names=['input_image'],    # 5. 언리얼 C++에서 접근할 입력 변수명
    output_names=['output_score'],  # 6. 언리얼 C++에서 접근할 출력 변수명
    dynamic_axes={                  # 7. (옵션) 0번 차원(배치)은 가변 개수로 열어두기
        'input_image': {0: 'batch_size'},    
        'output_score': {0: 'batch_size'}
    }
)