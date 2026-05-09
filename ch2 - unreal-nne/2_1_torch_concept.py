import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# torchvision.transforms : 완벽한 규격의 '사진' 만들기 공장

# 이미지 전처리
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), # 여기서 PIL 이미지가 PyTorch Tensor로 변환
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ])
# Resize(256): 사진의 짧은 쪽 길이를 256픽셀로 줄이거나 늘립니다
# CenterCrop(224): 정중앙을 기준으로 가로세로 224픽셀 크기의 정사각형으로 '가위질'
# ToTensor(): 픽셀(점)로 이루어진 이미지 파일을 AI가 읽을 수 있는 숫자 행렬(Tensor)로 교체. 이때 0~255 사이였던 RGB 색상 값을 0~1 사이의 소수점으로 스케일링.
# Normalize(...): 사진의 밝기와 대비를 '표준 상태'로 맞춤. 빛 번짐이 심하거나 너무 어두운 사진이라도, AI가 평소 공부했던 교과서 사진들과 비슷한 톤으로 보정해 주는 수학 연산

# model.eval(): AI의 뇌를 '학습 모드'에서 '실전 시험 모드'로 스위치 전환
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50( weights=weights)
model = model.to(device)
model.eval()
model.train() # 학습할 때 (언리얼에선 쓸 일 없음)
# AI 모델은 평소에 공부(Train)를 할 때, 틀리면 자기 뇌 구조(가중치)를 바꾸려고 함
# 만약 실전 테스트를 하는데 모델이 '어? 나 지금 공부하는 중인가?' 하고 착각해서 자기 뇌를 바꾸려고 하면 결과가 망가짐
# eval()은 Evaluate(평가)의 약자로, "이제 공부 끝! 뇌 모양 고정하고 내가 주는 문제에 답만 내놔!" 하고 잠금장치를 거는 아주 중요한 함수

# with torch.no_grad():
# 파이토치는 기본적으로 모델이 생각하는 모든 수학적 계산 과정을 메모리에 비디오처럼 녹화.
# 녹화 기능의 이름이 grad(기울기).
# 하지만 우리는 지금 녹화가 아닌 그냥 답만 필요한 '추론' 상황입. no_grad()는 "녹화하지 마! 메모리 아껴서 답만 빨리 내놔!"라고 지시하는것. 이 코드가 없으면 언리얼 엔진 같은 환경에서는 메모리가 터져버릴(OOM) 수 있다.
with torch.no_grad():
    # 이 들여쓰기 안에서 일어나는 모든 모델 연산은 녹화되지 않아 매우 빠르고 가볍습니다.
    output = model(...)

# torch.nn.functional.softmax: 중구난방인 점수를 '100점 만점 확률'로 변환하기
# AI가 뱉어낸 output을 보면 [12.5, -3.2, 5.1, ...] 처럼 자기 멋대로인 숫자들이 1000개 들어있다. 사람은 이 숫자를 보고 확신도를 알 수 없다.
# 소프트맥스(Softmax)라는 수학 공장에 이 숫자들을 넣으면, 1000개의 숫자를 모두 0에서 1 사이로 압축하고, 다 합치면 무조건 1(100%)이 되도록 예쁘게 비율을 맞춰준다
# 변환 전: [8.5, 1.2, -2.0]
# Softmax 변환 후: [0.98, 0.01, 0.01] -> "첫 번째가 정답일 확률 98%!"
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# torch.topk: 랭킹 리스트에서 'Top K' 뽑아내기
# 1000개의 확률값 중에서 우리가 궁금한 건 "그래서 제일 확률이 높은 게 뭔데?"
# topk(데이터, 1)이라고 쓰면, 전체 데이터 중에서 1등(Top 1)의 확률값(Score)과 그 1등이 몇 번째 자리에 있었는지 인덱스 번호(ID)를 동시에 반환.
# 만약 3등까지 알고 싶으면 topk(데이터, 3)을 쓰면 됨
top_prob, top_catid = torch.topk(probabilities, 1)
# top_prob: [0.98] (98% 확률이네!)
# top_catid: [208] (208번 카테고리구나!)

# .cpu().numpy() : 특수 작업장(GPU)에서 일반 사무실(CPU)로 박스 꺼내오기
# NumPy는 파이썬의 표준 수학 도구지만, GPU(그래픽카드) 메모리에 직접 들어갈 권한이 없다. 그래서 GPU에 있는 텐서(top_prob)를 읽으려고 하면 에러가 난다.
# 먼저 .cpu()를 써서 데이터를 컴퓨터의 기본 메모리(RAM)로 복사해 빼낸 다음, .numpy()를 써서 우리가 다루기 쉬운 기본 배열 객체로 포장을 바꾸는 필수 과정.
gpu_tensor = top_prob        # GPU에 있는 접근 불가 데이터
cpu_tensor = top_prob.cpu()  # RAM으로 꺼내옴
numpy_array = cpu_tensor.numpy() # 호환성 100% 넘파이 배열로 변환!

# weights.meta["categories"][top_catid] : 바코드 번호와 상품명 검색
# weights.meta["categories"]는 모델이 갖고 있는 일종의 바코드 사전(Dictionary/List)
# 이 사전에서 답인 모델의 이름을 출력하는것

# response.content: 날것의 데이터로 requests.get()으로 이미지를 요청하면, 하드디스크에 .jpg 파일로 예쁘게 저장되는 것이 아니라, 네트워크를 타고 온 순수한 바이트(Byte) 덩어리들이 메모리(RAM) 들어옴
# Image.open(): 원칙적으로 '파일(File)'을 열도록 설계되어 있어 하드디스크에 있는 C:/dog.jpg 같은 파일 경로나, 스트림(Stream) 객체를 달라고 요구한다. 그래서 메모리에 둥둥 떠 있는 순수 바이트 덩어리(response.content)를 그대로 밀어 넣으면 "이건 파일이 아닌데요?" 하면서 에러를 뱉는다
# BytesIO: 메모리를 파일처럼 속이는 역할을 하는 함수로 메모리(RAM)에 있는 바이트 덩어리를 감싸서 '읽고 쓸 수 있는 가상의 파일 객체(Memory Stream)'로 변신. 이렇게 포장하면 파일로 인식하여 이미지를 불어올 수 있다
url = "https://cdn.ecotiger.co.kr/news/photo/201211/4048_3074_309.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))