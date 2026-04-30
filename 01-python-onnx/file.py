#기본 방식
f = open("example.txt", "w") # 파일을 쓰기 모드로 연다
f.write("Hello, World!\n") # 파일에 문자열을 쓴다
f.close() # 파일을 닫는다

#다양한 방식
g= open("example.txt", "a") # 파일을 추가 모드로 연다
g.write("This is an additional line.\n") # 파일에 문자열을 추가로 쓴다
g.close() # 파일을 닫는다

with open("example.txt", "r") as f: # 파일을 읽기 모드로 연다
    content = f.read() # 파일의 내용을 읽는다
    print(content) # 읽은 내용을 출력한다

h = open(r"C:\Users\User\Documents\Git\ue5-story\example.txt2", "w") # 절대 경로로 파일을 연다
h.write("This file is created using an absolute path.\n") # 파일에 문자열을 쓴다
h.close() # 파일을 닫는다

for i in range(5):
    data = "Line %d\n" % i
    with open("example.txt", "a") as f: # 파일을 추가 모드로 연다
        f.write(data) # 파일에 문자열을 추가로 쓴다
# f.close() # 파일을 닫는다 (with 문을 사용하면 자동으로 닫힘)

# 라인 읽기
f = open("example.txt", "r") # 파일을 읽기 모드로 연다
line = f.readline() # 파일에서 한 줄을 읽는다
print(line) # 읽은 줄을 출력한다
f.close() # 파일을 닫는다

# 여러줄 읽기
f = open("example.txt", "r") # 파일을 읽기 모드로 연다
while True:
    line = f.readline() # 파일에서 한 줄을 읽는다
    if not line: # 더 이상 읽을 줄이 없으면 반복을 종료한다
        break
    print(line, end="") # 읽은 줄을 출력한다 (end 옵션으로 줄바꿈을 방지한다)
f.close() # 파일을 닫는다

#read 함수
f = open("example.txt", "r") # 파일을 읽기 모드로 연다
content = f.read() # 파일의 전체 내용을 읽는다
print(content) # 읽은 내용을 출력한다
f.close() # 파일을 닫는다