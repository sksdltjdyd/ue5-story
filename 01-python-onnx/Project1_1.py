import numpy as np
from PIL import Image

import glob

# 이미지 3개 불러오기
folder_path = 'C:/Users/seo/Documents/Git/ue5-story/01-python-onnx/Project_Img'

# 확장자가 jpg인 사진 3개 가져오기
orig_img = glob.glob(f'{folder_path}/*.jpg')[3:6]
print(orig_img)

num = 1

for i in orig_img:
    opened_img = Image.open(i)
    orig_array = np.array(opened_img)

    gray_img = opened_img.convert('L')
    gray_img_3channel = gray_img.convert('RGB')
    gray_array = np.array(gray_img_3channel)

    comb_array = np.hstack((orig_array, gray_array))

    final_img = Image.fromarray(comb_array)
    final_img.show()
    save_path = f'C:/Users/seo/Documents/Git/ue5-story/01-python-onnx/Project_Img/CombIMG_{num}.jpg'
    final_img.save(save_path)
    num+=1


