import numpy as np
from PIL import Image

# 이미지 불러오기
# Image.open() 함수가 내부적으로 _open() 함수를 호출하여 이미지를 열어서 Image 객체로 반환한다.
image = Image.open('test_image.jpg') # 'test_image.jpg'라는 파일에서 이미지를 불러온다. Image.open() 함수는 PIL 라이브러리에서 제공하는 함수로, 이미지 파일을 열어서 Image 객체로 반환한다. 'test_image.jpg'는 이미지 파일의 경로를 나타내며, 해당 파일이 존재해야 한다.

# 이미지 크기 출력
# image.size는 Image 객체의 size 속성을 나타낸다. size 속성은 이미지의 가로와 세로 크기를 튜플 형태로 반환한다. 예를 들어, (width, height) 형태로 반환된다. 따라서, print(image.size)는 이미지의 크기를 출력하는 코드이다.
print(image.size) # (width, height)
print(image.mode) # 이미지의 모드 출력 (예: 'RGB', 'L', 'RGBA' 등)
print(image.format) # 이미지의 포맷 출력 (예: 'JPEG', 'PNG' 등)

# 이미지 보여주기
image.show() 

# 배열 변환과 shape 출력
image_array = np.array(image) # Image 객체를 NumPy 배열로 변환한다. np.array() 함수는 입력된 객체를 NumPy 배열로 변환하는 함수이다. image는 Image 객체이므로, np.array(image)는 Image 객체를 NumPy 배열로 변환하여 반환한다. 이때, 이미지의 픽셀 데이터가 배열 형태로 저장된다.
print(image_array.shape) # 이미지 배열의 shape 출력 (예: (height, width, channels))
print(image_array.dtype) # 이미지 배열의 데이터 타입 출력 (예: uint8, float32 등)
print(image_array) # 이미지 배열의 픽셀 데이터 출력 (예: [[[R, G, B], [R, G, B], ...], [[R, G, B], [R, G, B], ...], ...])
print(image_array.ndim) # 이미지 배열의 차원 출력 (예: 3)

# 이미지 배열의 특정 픽셀 값 출력
print(image_array[0, 0]) # 이미지 배열의 (0, 0) 위치에 있는 픽셀 값 출력 (예: [R, G, B])
print(image_array[0, 0, 0]) # 이미지 배열의 (0, 0) 위치에 있는 픽셀의 R 값 출력 (예: R)
print(image_array[0, 0, 1]) # 이미지 배열의 (0, 0) 위치에 있는 픽셀의 G 값 출력 (예: G)
print(image_array[0, 0, 2]) # 이미지 배열의 (0, 0) 위치에 있는 픽셀의 B 값 출력 (예: B)

# image 자르기
# 이미지 배열의 영역을 자른다.
# image_array는 이미지 배열이므로, image_array[세로 구간, 가로구간]은 이미지 배열에서 세로구간의 특정 영역부터 특정영역까지 자른다는 말이고 가로 구간도 마찬가지다
# 이때, 자른 영역은 (height, width, channels) 형태의 배열이 된다.
cropped_image_array = image_array[0:2000, 0:2000]

 # 자른 이미지 배열을 Image 객체로 변환한다.
 # Image.fromarray() 함수는 NumPy 배열을 Image 객체로 변환하는 함수이다.
 # cropped_image_array는 자른 이미지 배열이므로, Image.fromarray(cropped_image_array)는 자른 이미지 배열을 Image 객체로 변환하여 반환한다.
cropped_img = Image.fromarray(cropped_image_array)

# 이미지 보여주기
cropped_img.show()

# 채널 분리해서 보여주기
red_channel = image_array[:, :, 0]
green_channel = image_array[:, :, 1]
blue_channel = image_array[:, :, 2]

Image.fromarray(red_channel).show()
Image.fromarray(green_channel).show()
Image.fromarray(blue_channel).show()

# 최종 이미지 저장하기
final_img = Image.fromarray(cropped_image_array)
final_img.save('result_cropped.jpg')
final_img.save('result_cropped.png')
print("이미지 저장 성공!")