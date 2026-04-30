# 클래스 기본
class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result
    
    def subtract(self, num):
        self.result -= num
        return self.result
    

calc = Calculator()
print(calc.add(5)) # 5
print(calc.add(3)) # 8
print(calc.subtract(2)) # 6

class FourCal:
    # 생성자 메서드
    # 객체가 생성될 때 자동으로 호출되는 메서드로, 객체의 초기화 작업을 수행한다
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def add(self):
        return self.first + self.second
    
    def subtract(self):
        return self.first - self.second
    
    def multiply(self):
        return self.first * self.second
    
    def divide(self):
        if self.second == 0:
            return "Cannot divide by zero"
        return self.first / self.second
    
calc = FourCal(3, 4)
print(calc.add()) # 7.0

# 상속
class MoreFourCal(FourCal):
    def pow(self):
        return self.first ** self.second

calc = MoreFourCal(3, 4)
print(calc.pow()) # 81.0

# 치환 메서드
class SafeFourCal(FourCal):
    def divide(self):
        if self.second == 0:
            return 0
        else:
            return self.first / self.second

calc = SafeFourCal(3, 4)
print(calc.divide()) # 0.75

# 클래스 변수와 인스턴스 변수
class Family:
    lastname = "Smith" # 클래스 변수

    def __init__(self, name):
        self.name = name # 인스턴스 변수

family1 = Family("John")
family2 = Family("Jane")
print(family1.lastname) # Smith
print(family2.lastname) # Smith
print(family1.name) # John
print(family2.name + " " + family2.lastname) # Jane Smith