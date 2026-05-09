import onnxruntime as ort

# 실습 목표 : ONNX Runtime을 이용해 모델이 요구하는 정확한 입출력 shape를 코드로 뽑아내 확인

# 모델 두개를 동시에 불러와서 비교
session_resnet = ort.InferenceSession("my_model.onnx")
session_mobile = ort.InferenceSession("mobilenet_unreal.onnx")

def print_model_info(session, model_name):
    print(f"{model_name} 모델 정보")

    #입력 정보 확인
    input_info = session.get_inputs()[0]
    print(f"[입력 핀] 이름: {input_info.name}")
    print(f"[입력 핀] 형태(Shape): {input_info.shape} -> 언리얼 C++ TArray 크기: {input_info.shape[1] * input_info.shape[2] * input_info.shape[3]}개")

    # 출력 정보 확인
    output_info = session.get_outputs()[0]
    print(f"[출력 핀] 이름: {output_info.name}")
    print(f"[출력 핀] 형태(Shape): {output_info.shape} -> 언리얼 C++ TArray 크기: {output_info.shape[1]}개")

print_model_info(session_resnet, "ResNet50")
print_model_info(session_mobile, "MobileNet V2")

"""
ResNet50 모델 정보
[입력 핀] 이름: input_image
[입력 핀] 형태(Shape): ['batch_size', 3, 224, 224] -> 언리얼 C++ TArray 크기: 150528개
[출력 핀] 이름: output_score
[출력 핀] 형태(Shape): ['batch_size', 1000] -> 언리얼 C++ TArray 크기: 1000개
MobileNet V2 모델 정보
[입력 핀] 이름: input_image
[입력 핀] 형태(Shape): [1, 3, 224, 224] -> 언리얼 C++ TArray 크기: 150528개
[출력 핀] 이름: output_image
[출력 핀] 형태(Shape): [1, 1000] -> 언리얼 C++ TArray 크기: 1000개
"""
