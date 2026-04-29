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

#return을 이용해 함수 종료하기
def say_nick(nick):
    if nick == "바보":
        return
    print("나의 별명은 %s입니다." % nick)

say_nick("천사") # 나의 별명은 천사입니다.
say_nick("바보") # 아무것도 출력되지 않는다

#매개변수 초기값 설정하기
def say_myself(name, old, man=True):
    print("나의 이름은 %s입니다." % name)
    print("나이는 %d살입니다." % old)
    if man:
        print("남성입니다.")
    else:
        print("여성입니다.")

say_myself("홍길동", 30) # 나의 이름은 홍길동입니다. 나이는 30살입니다. 남성입니다.
say_myself("홍길동", 30, False) # 나의 이름은 홍길동입니다. 나이는 30살입니다. 여성입니다

#전역변수 
c=1
def vartest(c):
    return c + 1

c = vartest(c)
print(c) # 2 --> vartest 함수는 c의 값을 1 증가시켜 반환한다

# 리스트, 딕셔너리, 튜플은 모두 참조형이므로 함수 안에서 값을 변경하면 함수 밖에서도 변경된다
a = [1, 2, 3]
def change_list(lst):
    lst.append(4)

change_list(a)
print(a) # [1, 2, 3, 4] --> change_list 함수는 a 리스트에 4를 추가한다


#람다함수 : 함수를 간단하게 표현할 수 있는 방법
add = lambda a, b: a + b
print(add(3, 4)) # 7