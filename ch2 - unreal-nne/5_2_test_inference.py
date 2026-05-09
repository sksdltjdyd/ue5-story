import numpy as np
from PIL import Image
import onnxruntime as ort
import requests      
from io import BytesIO
import json

# onnx 모델 로드
onnx_model_path = "my_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# 모델의 입력 소켓 이름 확인
input_name = session.get_inputs()[0].name

# 이미지 로드
"""
폴더에서 사진 로드
folder_path = 'C:/Users/seo/Documents/Git/ue5-story/01-python-onnx/Project_Img'
img_paths = glob.glob(f'{folder_path}/*.jpg')

if not img_paths:
    print("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    # 첫 번째 이미지로 테스트
    test_img_path = img_paths[0]
    opened_img = Image.open(test_img_path).convert('RGB')
    
    # 3. 데이터 전처리 (여권 사진 규격 맞추기)
    # ResNet50은 224x224 사이즈를 요구합니다.
    resized_img = opened_img.resize((224, 224))
    
    # 이미지를 NumPy 배열로 변환 (현재 HWC 형태: Height, Width, Channel)
    img_data = np.array(resized_img).astype(np.float32)
    
    # 0~255 픽셀 값을 0~1 사이로 정규화
    img_data = img_data / 255.0
    
    # HWC(높이,너비,채널) -> CHW(채널,높이,너비) 형태로 변환 (PyTorch/ONNX 기본 규격)
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # 배치 차원 추가 (CHW -> BCHW) : dummy_input이 (1, 3, 224, 224) 였음
    input_tensor = np.expand_dims(img_data, axis=0)

    # 4. ONNX Runtime으로 추론 실행!
    # session.run([결과받을출구이름], {입력소켓이름: 들어갈데이터})
    print("--- AI 모델이 이미지를 분석 중입니다 ---")
    outputs = session.run(None, {input_name: input_tensor})
    
    # 결과 배열 가져오기 (1000개의 클래스에 대한 점수)
    output_scores = outputs[0][0]
    
    # 5. 후처리 (Softmax 및 랭킹 1위 뽑기)
    # 가장 높은 점수를 가진 인덱스(1등) 추출
    best_class_idx = np.argmax(output_scores)
    
    print(f"가장 확률이 높은 클래스 ID (랭킹 1위): {best_class_idx}")
    print("성공적으로 추론을 마쳤습니다! 언리얼로 넘어가기 위한 완벽한 파이프라인입니다.")
"""

url = "https://cdn.ecotiger.co.kr/news/photo/201211/4048_3074_309.jpg" # 사진 인터넷에서 로드
response = requests.get(url) # 로드한 사진 불러오기
img = Image.open(BytesIO(response.content)).convert('RGB') # 불러온 이미지 오픈

# 데이터 전처리
resized_img = img.resize((224, 224))
img_data = np.array(resized_img).astype(np.float32)
img_data = img_data / 255.0
img_data = np.transpose(img_data, (2, 0, 1)) # HWC -> CHW
input_tensor = np.expand_dims(img_data, axis=0) # CHW -> BCHW

# onnx런타임 추론 실행
print("--- AI 모델이 이미지를 분석 중입니다 ---")
outputs = session.run(None, {input_name: input_tensor})
output_scores = outputs[0][0]

# 후처리
best_class_idx = np.argmax(output_scores)

print(f"가장 확률이 높은 클래스 ID (랭킹 1위): {best_class_idx}")

output_scores = outputs[0][0] # 모델이 뱉어낸 1000개의 원시 점수(Raw Score)

# 5. 후처리 (Softmax 및 랭킹 1위 뽑기)

# [Softmax 연산] 제각각인 원시 점수를 0~1 사이의 확률값(100점 만점 비율)으로 변환 
# (NumPy를 이용해 직접 수학 공식을 구현합니다)
exp_scores = np.exp(output_scores - np.max(output_scores))
probabilities = exp_scores / exp_scores.sum()

# [topk / argmax] 가장 확률이 높은 인덱스(ID) 획득 
best_class_idx = np.argmax(probabilities)
best_confidence = probabilities[best_class_idx] * 100 # %로 변환

# 6. 숫자 ID를 사람이 읽을 수 있는 문자로 매핑 (JSON 파싱)

# ImageNet 1000개 클래스 이름표가 있는 오픈소스 JSON URL
labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

print("--- 클래스 이름표(Dictionary)를 다운로드 중입니다 ---")
response_labels = requests.get(labels_url)
labels = response_labels.json() # JSON 데이터를 파이썬 리스트 형태로 변환

# 인덱스를 통해 영문 이름 매핑
predicted_label_en = labels[best_class_idx]

print("\n================ 결과 리포트 ================")
print(f"1. 랭킹 1위 클래스 ID : {best_class_idx}")
print(f"2. 인공지능의 예측 결과 : {predicted_label_en} (영문)")
print(f"3. 인공지능의 확신도    : {best_confidence:.2f}%")
print("===========================================\n")

# 3. 데이터 전처리 (여권 사진 규격 맞추기)
# 원본 비율을 무시하고 강제로 줄이는 버전 (CenterCrop까지 구현하면 너무 길어지므로 일단 Resize만 유지합니다)
resized_img = img.resize((224, 224))

# 이미지를 NumPy 배열로 변환 [H, W, C] 형태 [cite: 9]
img_data = np.array(resized_img).astype(np.float32)

# 1단계: 0~255 픽셀 값을 0~1 사이로 정규화 [cite: 12]
img_data = img_data / 255.0

# ---------------------------------------------------------
# 2단계: ImageNet 평균과 표준편차로 Normalize (추가된 부분!)
# ---------------------------------------------------------
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# 브로드캐스팅 마법: img_data는 (224, 224, 3) 형태이고 mean, std는 (3,) 형태입니다.
# NumPy가 알아서 R, G, B 채널 각각에 알맞은 값을 빼고 나누어줍니다!
img_data = (img_data - mean) / std
# ---------------------------------------------------------

# 3단계: HWC(높이,너비,채널) -> CHW(채널,높이,너비) 형태로 변환 [cite: 12]
img_data = np.transpose(img_data, (2, 0, 1))

# 4단계: 배치 차원 추가 (CHW -> BCHW)
input_tensor = np.expand_dims(img_data, axis=0)