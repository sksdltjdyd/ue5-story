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
    # 파일 경로(path)를 이용해 이미지를 열고, 'opened_img'라는 변수에 저장. 그레이 이미지로 전환
    opened_img = Image.open(i)
    gray_img = opened_img.convert('L')
    gray_img.show()

# 이미지 그레이로 바꾸기 두번째 방법
for i in img:
    # 원본 이미지 불러오고 NumPy 배열로 변환
    opened_img = Image.open(i)
    image_array = np.array(opened_img)

    # 채널 분리
    r = image_array[:, :, 0]
    g = image_array[:, :, 1]
    b = image_array[:, :, 2]

    # 공식 적용하기 (브로드캐스팅)
    gray_array = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # 데이터 타입 변환 (아주 중요)
    # 소수점을 곱했기 때문에 현재 gray_array 안의 숫자들은 소수점(float) 상태
    # 이미지는 0~255 사이의 정수여야 하므로, 정수형(uint8)으로 변환
    gray_array = gray_array.astype(np.uint8)

    # 다시 이미지 객체로 만들어서 확인
    gray_array_img = Image.fromarray(gray_array)
    gray_array_img.show()

num = 0
# 그레이 이미지 저장
for i in img:
    # 파일 경로(path)를 이용해 이미지를 열고, 'opened_img'라는 변수에 저장. 그레이 이미지로 전환
    opened_img = Image.open(i)
    gray_img = opened_img.convert('L')
    save_path = f'C:/Users/seo/Documents/Git/ue5-story/01-python-onnx/Project_Img/GrayIMG_{num}.jpg'
    gray_img.save(save_path)
    num += 1
  