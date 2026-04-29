def add(a, b):
    return a + b

print(add(1, 2))

#일반적인 함수
def add(a, b):
    result = a + b
    return result

#입력값이 없는 함수
def say_hello():
    return "Hello"

#출력값이 없는 함수
def print_hello(name):
    print("Hello my name is %s" % name)

a = print_hello("Alice")
print(a) # None --> 출력값이 없는 함수는 None을 반환한다

# 입력값과 출력값이 없는 함수
def print_hello_world():
    print("Hello World")

b = print_hello_world()
print(b) # None --> 입력값과 출력값이 없는 함수도 None을 반환한다

#매개변수 지정
def sub (a, b):
    return a - b

print(sub(5, 3)) # 2

result = sub(a=10, b=4)
print(result) # 6 

#매개변수가 몇 개가 될지 모르는 경우
def add_many(*args):
    result = 0
    for num in args:
        result += num
    return result

print(add_many(1, 2, 3, 4, 5)) # 15

#매개변수의 첫 번째 값이 연산 종류, 나머지 값이 숫자들인 경우
def add_mul(choice, *args):
    if choice == "add":
        result = 0
        for num in args:
            result += num
    elif choice == "mul":
        result = 1
        for num in args:
            result *= num
    return result

print(add_mul("add", 1, 2, 3, 4, 5)) # 15
print(add_mul("mul", 1, 2, 3, 4, 5)) # 120

#키워드 매개변수
def print_kwargs(**kwargs):
    print(kwargs)

print_kwargs(a=1, b=2, c=3) # {'a': 1, 'b': 2, 'c': 3}

#리턴값은 언제나 하나
def add_and_mul(a, b):
    return a + b, a * b

result = add_and_mul(3, 4)
print(result) # (7, 12) --> 튜플로 반환된다