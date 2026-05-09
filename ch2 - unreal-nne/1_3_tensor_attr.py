import numpy as np
import torch
# attribute : shape, dtype, device

#tensor.shape
a = torch.tensor([1,2,3,4,5])
a1 = a.shape
print(a1)

b = torch.ones([3,4,6])
b1 = b.shape
print(b1)

#tensor.shape
a2 = a.dtype
print(a2)

# 파이토치 자료형
float_tensor = torch.ones(1, dtype=torch.float)
double_tensor = torch.ones(1, dtype=torch.double)
int_tensor = torch.ones(1, dtype=torch.int)
long_tensor = torch.ones(1, dtype=torch.long)
bool_tensor = torch.ones(1, dtype=torch.bool)

print(float_tensor)
print(double_tensor)
print(int_tensor)
print(long_tensor)
print(bool_tensor)

# 현재 해당 텐서가 어느 디바이스에 있는지 확인
a3 = torch.tensor([1,2,3,4,5])
a_dev = a3.device
print(a_dev)

# 지금 GPU를 사용할 수 있는 환경인지 확인
can_gpu = torch.cuda.is_available()
print(can_gpu)

# GPU 이름 체크(cuda:0에 연결된 그래픽 카드 기준)
can_gpu = torch.cuda.get_device_name(device=0)
print(can_gpu)

# 사용 가능 GPU 개수 체크
can_gpu = torch.cuda.device_count()
print(can_gpu)

# 텐서를 GPU에 할당
a_gpu = torch.tensor([1.,2.,3.,4.,5.]).cuda()
b_gpu = torch.tensor([1.,2.,3.,4.,5.]).to("cuda")
print(a_gpu)
print(b_gpu)

a_dev = a_gpu.device
print(a_dev)