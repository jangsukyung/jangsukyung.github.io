---
layout: single
title:  "코딩 입문자를 위한 파이썬 기초 1"
categories: Lecture-Review
tag: [Python, study, Lecture, Startcoding, inflearn]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

# 1. 파이썬 다운로드 및 설치 하는 법 (윈도우)

## 파이썬 특징

- 문법이 정말 쉽고 간단하다.
- 인기가 많아서 공부할 자료가 많다
- 다양한 분야에서 활용된다.

## 파이썬 활용분야

- 크롤링
- 자동화
- 데이터분석
- 인공지능
- 웹서버개발(Django,Flask)
- 다양하게 사용 가능함.

# 2. Visual Studio Code 설치 및 사용법

## VS CODE란?

- 소스코드 편집기
- 프로그램을 작성하기 위한 편리한 기능들을 제공

## VS CODE 장점

- 폴더 및 파일 쉽게 정리
- 코드 자동완성
- 오류 수정이 쉽다
- 유용한 단축키가 많다

# 3. 코딩에서 가장 중요한 자료형과 변수 알아보기

- 자료형 → 자료의 형태, 데이터의 형태
- 숫자형 → 정수(integer), 실수(float)
- 문자형 → 문자열(String), ‘촉촉한 초코칩’, “상큼한 젤리”
- 변수 → 데이터를 저장할 공간, 데이터의 집
- 변수 생성 → 변수이름 = 데이터 (우변의 데이터를 변수이름에 저장)
- 예제

```python
name = "마스터이"
level = 5
health = 800
attack = 90

print(name, level, health, attack)

level = level + 1 # 5 + 1의 값을 level에 저장
health = 900 # 900의 값을 health에 저장
attack = 95 # 95의 값을 attack에 저장

print(name, level, health, attack)
```

# 4. 연산과 연산자를 겁나 쉽게 배우기

- 연산
    - 수나 식을 일정한 규칙에 따라 계산하는
- 연산의 종류
    - 대입연산, 산술연산, 비교연산, 논리연산
- 대입연산
    - name = “스타트코딩”
    - 등호 기호를 대입연산자라고 함
- 산술연산
    - +(더하기), -(빼기), *(곱하기), /(나누기), //(몫), %(나머지), **(제곱)
    - 숫자연산

```python
x = 5
y = 2

print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x//y)
print(x%y)
print(x**y)
```

- 문자열연산

```python
text1 = "#내꺼하자"
text2 = "#오늘부터1일"
text3 = "#내여친"

total_text = text1 + "\n" + text2 + "\n" + text3 * 3# \n은 Enter

print(total_text)
```

- 비교연산
    - >(크다), <(작다), > =(크거나 같다), < =(작거나 같다), ==(같다), ! = (다르다)
    - 불린형(Boolean)
    - True(참), False(거짓)
- 논리연산
    - A and B ⇒ A, B 모두 참이라면 True
    - A or B ⇒ A, B 중 하나라도 참이라면 Ture
    - not A ⇒ A가 참이라면 False

# 5. 입력과 자료형변환 간단하게 정리하기

- 입력
    - input() : 입력함수, 사용자로부터 데이터를 입력 받는 함수
- 에러메세지 = 선생님
- str ⇒ 문자열의 줄임말
- 문자열끼리 곱할 수 없음
- int(문자열) ⇒ 숫자형으로 변환
- operand : 연산자의 연산의 대상
    - x + y ⇒ +(연산자),  (x, y)(피연산자)

```python
year = int(input("태어난 연도를 입력해주세요 >>>"))
age = 2023 - year + 1

print(str(age) + "살입니다.")
```

# 6. 조건문 IF - 명령어 흐름을 제어해보자

- 제어문
    - 프로그램은 기본적으로 위에서 아래로 순차적으로 실행
    - 명령어A, 명령어B 중 한 개만 실행하기
    - 명령어A를 10번 반복하기
- 제어문의 종류
    - 조건문, 반복문
- 조건문
    - 조건에 따라서 실행할 명령이 달라진다.
    - if 뒤에 항상 한 칸 띄어쓰기 하기, 조건식(distance > = 200) → 비교연산

```python
if 조건식: # : <- 콜론, 명령블록이 시작된다는 의미
		조건식이 참일 때 실행되는 명령
# 들여쓰기 : 띄어쓰기 4칸, 탭(Tab)
else:
		조건식이 거짓일 때 실행되는 명령
```

```python
money = 9000

if money >= 20000:
    print("치킨과 맥주를 먹겠습니다.")
elif money >= 10000:
    print("떡볶이를 먹겠습니다.")
else:
    print("편의점 김밥행")
```

# 7. 조건문 예제

- 예제 1

```python
# 9만원이상 : 매도, 8 ~ 9만원 : 대기중, 8만원 미만 : 매수
price = int(input("삼성전자의 현재 가격을 입력해주세요. >>>"))

if price >= 90000:
    print("매도합니다.")
elif price >= 80000:
    print("대기중입니다.")
else:
    print("매수합니다.")
```

- 예제 2

```python
# 1. 사용자로부터 가방, 시계 금액 입력받기
# 2. 합계 금액 10만원 이상 할인율 30%, 5 ~ 10만원 할인율 20%, 5만원 미만 할인율 10%

bag_price = int(input("가방의 금액을 입력해주세요. >>>"))
watch_price = int(input("시계의 금액을 입력해주세요. >>>"))

total_price = bag_price + watch_price

if total_price >= 100000:
    total_price = total_price * 0.7
elif total_price >= 50000:
    total_price = total_price * 0.8
else:
    total_price = total_price * 0.9

print("합계 금액은 :", total_price)
```

# 8. 여러 개의 데이터를 저장할 수 있는 자료형, 리스트에 대해 알아보자

```python
# 리스트 생성하기
animals = ['사자', '호랑이', '고양이', '강아지']

# 데이터 접근하기
name = animals[0] # 0번째 인덱스

# 데이터 추가하기
animals.append('하마')
animals.append(1) # 같은 타입이 아니어도 가능

# 데이터 삭제하기
del animals[-1] # Delete, 마지막 데이터 삭제

# 리스트 슬라이싱
slicing = animals[1:3]

# 리스트 길이
length = len(animals)

# 리스트 정렬하기
animals.sort(reverse = True) # reverse : 내림차순

print(animals)
```

# 9. 프로그래밍의 꽃, 반복문 - for while 사용법 익히기

- for문

```python
for 변수 in 리스트:
		명령블록

for a in [1, 2, 3, 4]:
		print(a)
```

```python
names = ['티모', '리신', '이즈리얼']

for name in names:
    if name == '티모':
        print(name + "는 탑 챔피언입니다.")
    elif name == "리신":
        print(name + "은 정글 챔피언입니다.")
    elif name == "이즈리얼":
        print(name + "은 원딜 챔피언입니다.")
```

- range(10)
    - 0 ~ 9까지 순서열을 반환(순서열은 순서가 있는 데이터)
    - 정수를 입력 받아 순서열을 만들어주는 함수

```python
for i in range(60):
		print(i + 1, "분")
# 1분부터 60분까지 출력
```

```python
for i in range(12):
		print(i + 1, "월")
# 1월부터 12월까지 출력
```

- range(1, 10)
    - 1부터 9까지의 숫자

```python
for i in range(1, 11):
    print(i, "번째 페이지입니다.")
```

- range(1, 10, 2)
    - (시작숫자, 끝숫자 + 1, 단계)

- while문

```python
count = 0
while count < 5:
		print(count, "번째 반복입니다.")
		count = count + 1

while 조건:
		명령블록
```

- for문, while문
    - for문 ⇒ 정한 횟수만큼 반복
    - while문 ⇒ 조건을 만족하지 않을 때까지 반복

# 10. 파이썬 반복문 예제 3문제

- 예제1

```python
num = int(input("자연수를 하나 입력해주세요. >>>"))

sum = 0

for i in range(1, num + 1):
    sum = sum + i

print(sum)
```

- 예제2

```python
print("프로그램 시작")

# num = int(input("종료하려면 -1을 입력하세요:"))

# while num != -1:
#     num = int(input("종료하려면 -1을 입력하세요:"))

while True: # 계속 반복을 하겠습니다.
    num = int(input("종료하려면 -1을 입력하세요."))
    if num == -1:
        break

print("프로그램 종료")
```

- 예제3

```python
while True:
    print("메뉴를 입력하세요.")
    select = int(input("1. 게임시작 2. 랭킹보기 3. 게임종료 >>>"))

    if select == 1:
        print("게임을 시작합니다.")
    elif select == 2:
        print("-> 랭킹보기")
    elif select == 3:
        print("-> 게임을 종료합니다.")
        break
    else:
        print("-> 다시 입력해 주세요.")
```

# 11. 반복문 연습 예제 - 별찍기

- 별찍기1

```python
for i in range(1, 6): # 1, 2, 3, 4, 5
    print('*' * i)
```

- 별찍기2

```python
for i in range(1, 6):
    print('*' * (6 - i))
```

- 별찍기3

```python
for i in range(1,6):
    print(' ' * (5 - i) + '*' * i)
```

- 별찍기4

```python
for i in range(1, 6):
    print(' ' * (i - 1) + '*' * (6 - i))
```