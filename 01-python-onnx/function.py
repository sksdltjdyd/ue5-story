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
print(a) # None --> 출력값이 없는 함수는 None을 반환한다.