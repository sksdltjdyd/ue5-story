import numpy as np
import torch

# 텐서의 shpae, dimension을 조절하는 방법
# 여러개의 텐서를 합치는 방법

# tensor.reshape()
# 모양 변경. 원래 사이즈를 구성하는 각 차원 별 길이의 곱으로 표현 가능하다면 변경 가능
# tensor.shape이[x, y]라면, 총 input size인 x*y의 값으로 표현할 수 있는 차원의 조합이면 가능

a = torch.tensor([[1,2,3,4], [5,6,7,8]])

print(a.shape)
print(a.reshape([4,2]))
print(a.reshape([1,2,4]))
print(a.reshape([2,2,2]))
print(a.reshape([8]))
# print(a.reshape([3,4])) --> x*y값이 기존 shape값의 x*y를 벗어나기 때문에 불가능

# tensor.squeez()
# 차원의 값이 1인 차원을 제거한다
# 차원의 값이 1인 차원이 여러개인 경우, 차원을 지정하면 해당 차원만 제거한다

a = torch.rand([3,1,3])
print(a)
print(a.shape)
print(a.squeeze())
print(a.squeeze().shape)

b = torch.rand(1,1,2,128)
print(b.shape)
print(b.squeeze(dim=1).shape) # [1, 2, 128]

# tensor.unsqueeze()
# 반대로 1인 차원을 생성하는 함수
# 그래서 어느차원에 1인 차원을 생성할지 꼭 지정해줘야 한다
c = torch.tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(c)
print(c.shape)
print(c.unsqueeze(dim=0))
print(c.unsqueeze(dim=0).shape)
print(c.unsqueeze(dim=1))
print(c.unsqueeze(dim=1).shape)
print(c.unsqueeze(dim=2))
print(c.unsqueeze(dim=2).shape)

# torch.cat()
# 쉽게 두개의 텐서를 합치는 작업
# 합치려는 차원을 제외한 나머지 차원의 경우 두 텐서의 모양이 같아야 한다
d = torch.rand(5, 24, 50) # [m,n,k]
e = torch.rand(5, 24, 50) # [m,n,k]

out1 = torch.cat([d,e], dim=0) # [m+m,n,k]
print(out1.shape)
out2 = torch.cat([d,e], dim=1) # [m,n+n,k]
print(out2.shape)
out3 = torch.cat([d,e], dim=2) # [m,n,k+k]
print(out3.shape)

# tensor.view(-1) 와 tensor.flatten(): 3차원 배열을 일렬로 쭉 이어붙이는 작업
# 언리얼 C++로 데이터를 넘길 때는 다차원 배열 [3, 224, 224]를 쓸 수 없고, 무조건 한 줄로 된 긴 기차 [150528] 형태로 만들어야 함. 이때 쓰는 도구들로 두 기능은 사실상 똑같은 역할
# tensor.view(-1): view는 텐서의 모양을 바꾸는 함수. 여기서 -1은 파이토치에게 "네가 알아서 남은 숫자들 다 곱해서 한 줄로 쫙 펴!"라고 계산을 맡기는 마법의 숫자. 실무에서 정말 많이 쓴다
# tensor.flatten(): 직관적인 이름 그대로 "납작하게 찌그러트려라!" 명령하는 함수
tensor_3d = torch.randn(3, 224, 224) # 3D 텐서
flat_1 = tensor_3d.flatten() # 1차원으로 펴짐
flat_2 = tensor_3d.view(-1)  # 똑같이 1차원으로 펴짐 (-1이 자동 계산)
print(flat_1)
print(flat_2)