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