import onnxruntime as ort

'''
# CPU 버전
print(f"ONNX Runtime 버전: {ort.__version__}")
print(f"사용 가능한 실행 제공자: {ort.get_available_providers()}")
'''

# GPU 테스트
# 사용할수 있는 가속기 가져오기
prov = ort.get_available_providers()
print(f"사용 가능 가속기 : {prov}")

if 'CUDAExecutionProvider' in prov:
    print("CUDA 사용 가능")
else:
    print("CUDA 사용 불가능")


