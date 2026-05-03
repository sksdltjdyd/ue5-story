import numpy as np
from PIL import Image

import glob

# 이미지 3개 불러오기
folder_path = 'C:/Users/seo/Documents/Git/ue5-story/01-python-onnx/Project_Img'

# 확장자가 jpg인 사진 3개 가져오기
img = glob.glob(f'{folder_path}/*.jpg')[:3]
print(img)

# 이미지 3개 띄우고 정보 출력하기
for i in img:
    # 파일 경로(path)를 이용해 이미지를 열고, 'opened_img'라는 변수에 저장
    opened_img = Image.open(i)

    # 이미지 띄우기
    opened_img.show()

    print(f"--- {img} 정보 ---")
    print(opened_img.size) # (width, height)
    print(opened_img.mode) # 이미지의 모드 출력 (예: 'RGB', 'L', 'RGBA' 등)
    print(opened_img.format) # 이미지의 포맷 출력 (예: 'JPEG', 'PNG' 등)
    print("-" * 30)

for i in img:
    # 파일 경로(path)를 이용해 이미지를 열고, 'opened_img'라는 변수에 저장
    opened_img = Image.open(i)
    gray_img = opened_img.convert('L')
    gray_img.show()