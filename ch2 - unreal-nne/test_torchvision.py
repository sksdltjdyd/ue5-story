import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# torchvision.models: 똑똑한 인공지능 '전문가'들을 모아둔 곳. 여기서 사진 속 사물을 분류하는 데 특화된 ResNet50을 사용
# torchvision.transforms: 현실의 사진(이미지 파일)을 AI가 이해할 수 있는 숫자 행렬(Tensor)로, 그것도 AI가 딱 좋아하는 사이즈(224x224)와 색상 규격으로 가공

# GPU 사용을 위한 세팅
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용이 가능하다면 쿠다 사용, 그게 아니면 cpu 사용
print(f"현재 세팅: {device}")

# 사전 학습 모델 로드 및 설정
# 최신 가중치(학습된 데이터)를 포함하여 ResNet50 모델 로드
weights = models.ResNet50_Weights.DEFAULT # 버전 선택 및 로드 / 파이토치는 모델을 업데이트하면서 성능이 더 좋아진 가중치 V1, V2 등을 계속 내놓고 있다. DEFAULT라고 쓰면 "현재 파이토치 버전에서 가장 성능이 좋은 최신 가중치 파일로 줘!"라는 뜻
model = models.resnet50( weights=weights) # ResNet50이라는 신경망 구조를 메모리에 생성하고, 아까 위에서 고른 가중치(weights)를 그 신경망 안에 덮어씌우는 코드

# 모델을 GPU 메모리로 이동시키고 추론 모드로 변경
model = model.to(device)
model.eval()
print("모델 로드 및 평가 모든 전환")

# 이미지 로드
url = "https://cdn.ecotiger.co.kr/news/photo/201211/4048_3074_309.jpg" # 사진 인터넷에서 로드
response = requests.get(url) # 로드한 사진 불러오기
img = Image.open(BytesIO(response.content)) # 불러온 이미지 오픈

# 이미지 전처리
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), # 여기서 PIL 이미지가 PyTorch Tensor로 변환
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ])

# 이미지 전처리 실행(결과 : [3, 224, 224]형태의 텐서)
in_tensor = preprocess(img)

# 모델은 항상 묶음(Beta) 단위를 원하므로 차원을 하나 추가: [1, 3, 224, 224]
# 중요포인트) 입력 데이터도 모델이 있는 장비로 이동
in_batch = in_tensor.unsqueeze(0).to(device)
print(f"입력 텐서 모양: {in_batch.shape}")

# 추론단계
# 모델 안의 기울기 계산 엔진을 써서 속도와 메모리 효율을 높임
with torch.no_grad():
    output = model(in_batch)

print(f"출력 텐서 모양: {output.shape}") # [1, 1000] (1000개의 카테고리 접수)

# Numpy변환
# 1000개의 점수를 확률(0~1)로 변환
probabil = torch.nn.functional.softmax(output[0], dim=0)

# 가장 확률이 높은 1개의 카테고리 ID와 그 확률값 추출
top_prob, top_catid = torch.topk(probabil, 1)

# 핵심) 텐서를 Numpy로 변환
# 만약 텐서가 GPU에 있다면, 무조건 .cpu()로 메모리를 내린 후에 .numpy()를 호출해야함!
prob_num = top_prob.cpu().numpy()

# 결과 출력
category_name = weights.meta["categories"][top_catid]
print(f"이 사진은 {prob_num[0] * 100:.2f}% 확률로 {category_name}입니다")