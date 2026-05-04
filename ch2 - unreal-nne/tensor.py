import numpy as np
import torch

# 텐서 다루기 실습

# 텐서 만들기
a = torch.tensor(5)
print(a)
b = torch.tensor([1,2,3,4,5])
print(b)
c = torch.tensor([[1],[2],[3],[4],[5]])
print(c)

# list나 set, numpy array등에도 적용 가능
temp_list = [1,2,3,4,5]
d = torch.tensor(temp_list)
print(d)

temp_set = (1,2,3,4,5)
e = torch.tensor(temp_set)

temp_array = np.array(temp_list)
f = torch.tensor(temp_array)
print(f)

g = torch.from_numpy(temp_array)
print(g)

# torch.ones()를 통해 만들면 각 element가 1인 텐서 만들어줌
test1 = torch.ones(3)
print(test1)

test2 = torch.ones([3,3])
print(test2)

test3 = torch.ones([3,3,3])
print(test3)

# torch.zeros()를 통해 만들면 각 element가 0인 텐서를 만들어줌
# Argument로는 텐서의 dimension을 입력
z1 = torch.zeros(3)
print(z1)

z2 = torch.zeros([2, 3])
print(z2)

z3 = torch.zeros([4,5,3])
print(z3)

# torch.arange()를 통해 주어진 범위 내의 정수를 순서대로 생성
a1 = torch.arange(1, 10)
print(a1)

# 22부터 10까지 -1씩 작아지며 순서대로 생성
a2 = torch.arange(22, 10, -1)
print(a2)

# torch.rand()
# [0, 1]의 uniform dist에서 난수를 생성해서 텐서로 만들어줌

# 0~1 사이 난수를 5개 만들어서 텐서로 만들어줌
r1 = torch.rand(5)
print(r1)

# 0~1 사이 난수를 5개 만들고 그런 집단을 4줄 만들어서 텐서로 생성
r2 = torch.rand([4,5])
print(r2)

# 0~1 사이 난수를 2개 만들고 그런 집단을 3줄씩 총 3개 만들어서 텐서로 생성
r3 = torch.rand([3,3,2])
print(r3)

# torch.xxx_like()를 통해 기존의 텐서와 같은 모양의 원소가 1인 텐서를 만들어줌
l1 = torch.zeros([3,2,4])
torch.ones_like(l1)
print(l1)