import numpy as np
from PIL import Image
import onnxruntime as ort
import requests

# 올라갈 AI 모델 세팅
print("1. ONNX 모델 로드중")
onnx_model_path = "mobilenet_unreal.onnx" # 만든 onnx 로드
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

# 커스텀 데이터 준비
print("2. Image 로드중")
img_path = 'C:/Users/seo/Documents/Git/ue5-story/ch1 - python-onnx/Project_Img/CombIMG_3.jpg'
# 흑백 이미지 3개채널로 강제 확장
opened_img = Image.open(img_path).convert('RGB')

# 데이터 전처리(C++ 배열화 및 정규분포화)
print("3. AI가 이미지를 전처리중")
resized_img = opened_img.resize((224, 224))
img_data = np.array(resized_img).astype(np.float32)

# 정규화 1단계 0~255픽셀 --> 0~1 압축
img_data = img_data / 255.0

# 정규화 2단계 색감 조정
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_data = (img_data - mean) / std

# HWC -> CHW 변환
img_data = np.transpose(img_data, (2, 0, 1))

# 배치 차원 추가 (CHW -> BCHW)
# 언리얼 C++로 치면 150,528칸짜리 배열이 완성된 상태!
input_tensor = np.expand_dims(img_data, axis=0).astype(np.float32)

# ONNX Runtime 추론 실행
print("4. ONNX 추론 엔진 가동")
outputs = session.run(None, {input_name: input_tensor})
output_scores = outputs[0][0] # 1000개의 점수가 담긴 배열 반환

# 후처리
print("5. 결과 분석 중...")
# Softmax 수학 공식: 제각각인 점수를 0~1 사이의 확률값(100점 만점)으로 변환
exp_scores = np.exp(output_scores - np.max(output_scores))
probabilities = exp_scores / exp_scores.sum()

# 가장 높은 확률을 가진 1등 인덱스와 그 퍼센티지(%) 추출
best_class_idx = np.argmax(probabilities)
best_confidence = probabilities[best_class_idx] * 100

# 1000개 클래스 영문 이름표 다운로드 및 매핑
labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response_labels = requests.get(labels_url)
labels = response_labels.json()
predicted_label_en = labels[best_class_idx]

# 최종 결과 출력
print("\n================ W2 최종 결과 리포트 ================")
print(f"이미지 경로   : {img_path}")
print(f"랭킹 1위 ID   : {best_class_idx}")
print(f"인공지능 예측 : {predicted_label_en}")
print(f"확신도(확률)  : {best_confidence:.2f}%")
print("=====================================================\n")
print("🎉 축하합니다! AI 추론 파이프라인의 블랙박스를 완벽히 깨셨습니다! 🎉")