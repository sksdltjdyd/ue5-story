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
결과:
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

'''
1. 매직 넘버 150528과 1000
두 모델 모두 내부의 뇌 구조(파라미터 개수)는 다르지만, 입출력 규격(Shape)은 완전히 동일하다는 것을 확인
    - 입력 핀 (150528개):
        - 모델은 [3, 224, 224] 형태의 데이터를 받는다. 이는 3채널(RGB) * 224(가로) * 224(세로) 픽셀을 의미
        - 언리얼 C++의 TArray는 다차원 구조를 이해하지 못하므로 데이터를 1차원으로 펴서 밀어넣어야 한다. 즉, 이미지 1장을 밀어 넣으려면 언리얼에서 정확히 TArray<float> 150,528칸짜리 빈 배열을 준비해야 한다는 의미
    - 출력 핀 (1000개):
        - 모델이 분석을 마치고 뱉어내는 데이터
        - ImageNet의 1,000개 사물에 대한 점수표이므로, 결과를 받을 때도 언리얼에서 TArray<float> 1,000칸짜리 빈 배열을 대기시켜 놓고 결과값을 받아와야 한다
2. batch_size vs 1의 의미
맨 앞의 0번 차원(배치 차원)에 적힌 글자가 다른데, 이 부분이 바로 가장 중요한 차이점
    - ResNet50 ('batch_size'):
        - test_onnxexport.py 코드를 보면, 맨 밑에 dynamic_axes={'input_image': {0: 'batch_size'}}라는 옵션 존재
        - 이는 TArray처럼 입력 데이터의 개수(배치)를 가변적으로 허용하겠다는 뜻
        - 즉, 사진을 1장 넣어도 되고, 10장을 한 번에 넣어도 처리할 수 있게 "입구를 고무줄처럼 늘어나게" 만들어 둔 것
    - MobileNet V2 (1):
        - dynamic_axes 옵션이 존재하지 않음
        - 입구가 무조건 1장만 받도록 숫자 1로 굳어버린(Bake) 것
3. 언리얼 NNE 파이프라인 최적화 전략
언리얼 엔진에 이 모델들을 올릴 때, C++ 코드로 메모리를 할당하는 과정은 다음과 같이 이루어진다
    - 메모리 할당: TArray<float> InputData; InputData.SetNum(150528);
    - 데이터 채우기: 카메라나 이미지 파일에서 픽셀을 읽어와서 InputData의 150,528칸을 가득 채운다. (이때 NumPy에서 했던 0~1 정규화 연산을 C++로 해준다)
    - 추론 실행: 엔진(NNE)에 InputData를 밀어 넣는다
    - 결과 받기: TArray<float> OutputData; OutputData.SetNum(1000); 에 1,000개의 점수를 받아온다
'''