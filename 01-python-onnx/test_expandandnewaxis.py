import numpy as np

# 넘파이 배열 축 추가하기
a = np.array([1, 2, 3, 4, 5])
print("Original array:", a)
print("Original shape:", a.shape)

# np.expand_dims()를 사용하여 새로운 축 추가
# axis=0은 첫 번째 차원에 새로운 축을 추가하는 것을 의미한다. 따라서, (5,)인 배열이 (1, 5)로 바뀌게 된다.
a_expanded = np.expand_dims(a, axis=0)
print("After expand_dims(axis=0):", a_expanded)
print("New shape:", a_expanded.shape)

# axis=1은 두 번째 차원에 새로운 축을 추가하는 것을 의미한다. 따라서, (5,)인 배열이 (5, 1)로 바뀌게 된다.
a_expanded = np.expand_dims(a, axis=1)
print("After expand_dims(axis=1):", a_expanded)
print("New shape:", a_expanded.shape)

# np.newaxis를 사용하여 새로운 축 추가
# np.newaxis는 np.expand_dims()와 동일한 기능을 하는 축 추가 방법이다. np.newaxis를 사용하여 새로운 축을 추가할 때는, 배열의 인덱싱에서 np.newaxis를 사용하여 원하는 위치에 새로운 축을 추가할 수 있다.
# 예시에서 a[:, np.newaxis]는 a의 모든 요소에 대해 새로운 축을 추가하는 것을 의미한다. 따라서, (5,)인 배열이 (5, 1)로 바뀌게 된다.
a_newaxis = a[np.newaxis, :]
print("After np.newaxis (axis=0):", a_newaxis)
print("New shape:", a_newaxis.shape)

# np.newaxis를 사용하여 새로운 축을 추가할 때, a[:, np.newaxis]는 a의 모든 요소에 대해 새로운 축을 추가하는 것을 의미한다. 따라서, (5,)인 배열이 (5, 1)로 바뀌게 된다.
a_newaxis = a[:, np.newaxis]
print("After np.newaxis (axis=1):", a_newaxis)
print("New shape:", a_newaxis.shape)