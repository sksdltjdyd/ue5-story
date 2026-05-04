# 기본방식. input() 함수로 입력 받는다
a = input()
print("You entered:" + a)

# 입력받을 때 안내문구를 띄우는 방식
b = input("Enter something: ")
print("You entered:" + b)

for i in range(3):
    print(i, end=" ") # end 옵션을 이용하여 줄바꿈 대신 공백으로 구분하여 출력한다