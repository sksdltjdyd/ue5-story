import numpy as np
import torch

# tensor 연산, min, max, mean

# element-wise 연산
# 텐서의 같은 자리 원소들끼리 산술 연산

a = torch.tensor([[3,3],[4,5]])
b = torch.tensor([[5,6],[5,3]])

print(a)
print(b)
print(a + b)
print(a - b)
print(a * b)
print(a / b)


print(a.add(b))
print(a.sub(b))
print(a.mul(b))
print(a.div(b))

# inplace 연산
# 함수 뒤에 _를 붙이면 자기 자신의 값을 바꾸게 된다
# 연산 결과를 반환하면서 동시에 자기 자신의 데이터를 고침
print(a.add(b))
print(a)
print(a.add_(b))
print(a)


# tensor.min()
# 가장 작은 값을 반환
# argument로 차원을 넣어주면 해당 차원에서 각각 작은 값들과, 그 값들의 위치를 함께 리턴
c_min = torch.tensor([[3,1,4], [2,4,3]])
print(c_min.shape)
print(c_min.min())

# dim=0 : 위 아래 세로 축 방향
# 그래서 다른 층끼리 경쟁
# values=tensor([2, 1, 3]) / indices=tensor([1, 0, 1]))
print(c_min.min(dim=0))

# dim=1 : 가로축 방향
# 같은 층에서 크게 경쟁
# values=tensor([1, 2]) / indices=tensor([1, 0]))
print(c_min.min(dim=1))

# 3차원 텐서
d_min = torch.tensor([[[3,4,5], [1,2,3]], [[4,5,6], [7,8,4]]])
print(d_min)
print(d_min.shape)

# 서로 다른 집합끼리 비교
# values=tensor([[3, 4, 5], [1, 2, 3]]),
# indices=tensor([[0, 0, 0], [0, 0, 0]]))
print(d_min.min(dim=0))

# 같은 집합내 세로축 비교
# values=tensor([[1, 2, 3], [4, 5, 4]]),
# indices=tensor([[1, 1, 1], [0, 0, 1]]))
print(d_min.min(dim=1))

# 같은 집합내 가로축 비교
# alues=tensor([[3, 1], [4, 4]]),
# indices=tensor([[0, 0], [0, 2]]))
print(d_min.min(dim=2))

# tensor.max
# 가장 큰 값을 반환
# argument로 차원을 넣어주면 해당 차원에서 각각 큰 값들과, 그 값들의 위치를 함께 리턴
c_max = torch.tensor([[3,1,4], [2,4,3]])
print(c_max.shape)
print(c_max.max())

# dim=0 : 위 아래 세로 축 방향
# 그래서 다른 층끼리 경쟁
print(c_max.max(dim=0))

# dim=1 : 가로축 방향
# 같은 층에서 크게 경쟁
print(c_max.max(dim=1))

# 3차원 텐서
d_max = torch.tensor([[[3,4,5], [1,2,3]], [[4,5,6], [7,8,4]]])
print(d_max)
print(d_max.shape)

# 서로 다른 집합끼리 비교
print(d_max.max(dim=0))

# 같은 집합내 세로축 비교
print(d_max.max(dim=1))

# 같은 집합내 가로축 비교
print(d_max.max(dim=2))

# tensor.mean()
# 텐서 전체 평균 혹시 차원별 평균값 리턴
c_mean = torch.FloatTensor([[1,2,3], [5,6,8]])
print(c_mean)
print(c_mean.shape)
print(c_mean.mean(dim=0))
print(c_mean.mean(dim=1))

# Indexing
# 원하는 원소나 영역을 index를 이용하여 선택
a_ind = torch.tensor([[1,2,3], [4,5,6], [7, 8, 9]])
print(a_ind.shape)
print(a_ind[0][2])

b_ind = torch.rand([3,3,3])
print(b_ind)
print(b_ind[0][2][0])