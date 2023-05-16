---
layout: single
title:  "핵심만 빠르게, 입문자를 위한 파이썬(Python)과 판다스(Pandas) 2"
categories: Lecture-Review
tag: [Python, study, Lecture,inflearn, pandas]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

# 1. DataFrame의 수정 방법

- DataFrame 수정 (update)
    - 새로운 열 추가
    - 데이터 수정

- 새로운 열 추가

```python
sample['인구밀집지'] = True
```

- 데이터 수정
- 생활인구 합계가 1000 이상인 곳만 인구 밀집지로 지정하자.

```python
sample['생활인구합계'] >= 1000
```

```python
sample['인구밀집지'] = sample['생활인구합계'] >= 1000
sample
```

- 첫번째 행에 있는 행정동 코드를 임의로 수정해보자.

```python
sample.iloc[0, 1] = '100'
sample.head()
```

- 사실 수정은.. 조회 문법에 대입연산자(=)만 넣으면 끝이다.

# 2. DataFrame의 삭제 방법

- DataFrame 삭제 (delete)
- 컬럼 삭제
    - axis = 0 (행), axis = 1 (열)

```python
sample.drop(['인구밀집지', '등록일자'], axis = 1)
```

- 로우 삭제

```python
sample.drop(20170226, axis = 0)
```

- 판다스는 데이터 보호를 위해 drop 메소드를 사용했다고 해서, 원본 데이터를 삭제하진 않는다.

```python
droppedSample = sample.drop(["인구밀집지", "등록일자"], axis = 1)
droppedSample
```

# 3. Pandas의 연산 기능 (sum, divide, cumulative)

- 자주 사용하는 pandas 연산 기능 함수 소개

```python
df = pd.DataFrame({
	"가로" : [10, 20, 30, 10, 30, 20, 11],
	"세로" : [20, 23, 22, 33, 22, 12, 11],
	"높이" : [50, 40, 20, 50, 20, 30, 40]
})

df
```

- sum : 합계

```python
# sum 합계 계산 axis = 0
df.sum(axis = 0) # default
```

```python
# sum 합계 계산 axis = 1
df.sum(axis = 1)
```

- divide : 나누기

```python
df.divide(2) # 나눌 숫자
```

```python
# sum과 함께 응용하자면 .. 이런식으로도 사용 가능
df.divide(df.sum(axis = 1), axis = 0)
```

- 누적 계산 (누적곱, 누적 최대, 최솟값)

```python
# cumprod 누적곱
df.cumprod(axis = 0)
```

```python
# cumprod 누적 최댓값, 최솟값
df.cummax()
df.cummin()
```

# 4. DataFrame에 파이썬 함수를 적용할 수 있는 Apply 함수

- apply 함수
    - dataframe에 파이썬 함수를 적용할 수 있다.
    - 예를 들어, 가로, 세로, 높이를 이용해 부피라는 컬럼을 추가해보자.

```python
def getVolume(row):
	return row['가로'] * row['세로'] * row['높이']

df['부피'] = df.apply(getVolume, axis = 1)
df
```

# 5. 서로 다른 DataFrame을 합칠 수 있는 Concat 함수

- concat 함수
    - 서로 다른 두 개의 데이터 프레임을 합치는 기능

```python
df2 = pd.DataFrame({
	"가로" : [10, 20, 30, 10, 30, 20, 11],
	"세로" : [20, 23, 22, 33, 22, 12, 11],
	"높이" : [50, 40, 20, 50, 20, 30, 40]
})

df2
```

```python
pd.concat([df, df2], axis = 0) # axis = 0 이 default
```

```python
pd.concat([df, df2], axis = 1)
```

# 6. 중복 데이터를 처리하는 방법

- 중복 데이터 찾기 : duplicated

```python
df = pd.DataFrame({
	'brand' : ['Yum Yum', 'Yum Yum', 'indomie', 'indomie', 'indomie'],
	'style' : ['cup', 'cup', 'cup', 'pack', 'pack'],
	'rating' : [4, 4, 3.5, 15, 5]
})
df
```

```python
df.duplicated()
```

```python
# keep이라는 속성을 사용하여 어떤 값을 중복으로 인식할 것인지 설정, 기본값은 first
df.duplicated(keep = 'last')
```

```python
# subset 속성을 사용하여 특정 컬럼에 대한 중복만 처리 가능
df.duplicated(subset = ['brand'])
```

- 중복값 삭제 : drop_duplicates

```python
df.drop_duplicates()
```

```python
df.drop_duplicates(subset = ['brand', 'style'], keep = 'last')
```

# 7. 결측 데이터를 처리하는 방법

- 결측(빈) 데이터 처리
- isnull (isna) : 결측치 찾기

```python
import numpy as np
```

```python
df = pd.DataFrame([[1, 5, 2], [np.nan, 1, 3], [6, np.nan, 20], [2, 5, np.nan], [2, 5, 2]])
df
```

```python
df.isnull()
```

```python
df.isnull().sum()
```

- fillna : 결측치 채우기

```python
df.fillna(1) # 특정 값으로 채우기
```

```python
df.fillna(method = 'ffill') # 특정 값으로 채우기 ffill = forward fill 앞에 값으로 대체
```

```python
df.fillna(method = 'bfill') # 특정 값으로 채우기 bfill = backward fill 뒤에 값으로 대체
```

- dropna : 결측치 삭제

```python
df = pd.DataFrame([[1, 5, 2], [np.nan, 1, 3], [6, np.nan, 20], [2, 5, np.nan], [2, 5, 2]])
df
```

```python
df.dropna()
```

# 8. 간단한 시각화 방법 소개

- 시각화
    - matplotlib를 사용하여 기본적인 그래프를 만들 수 있다.
    - 라인 플롯 (line plot)
    - 스케터 플롯 (scatter plot)
    - 컨투어 플롯 (contour plot)
    - 서피스 플롯 (surface plot)
    - 바 차트 (bar chart)
    - 히스토그램 (histogram)
    - 박스 플롯 (box plot)
- 등 다양한 그래프를 만들 수 있는데, 사실 사용법은 필요에 따라 검색해서 그때 그때 익히면 된다.
- 여기서는 라인 플롯을 기준으로 그래프 사이즈, 레이블 설정 등 기본적인 사용법을 위주로 살펴본다.

```python
import matplotlib.pyplot as plt
```

```python
sample = pd.read_csv('내국인 생활인구.csv', encoding = 'cp949')
sample.head()
```

```python
# 윈도우
plt.rc('font', family = 'Malgun Gothic')

sample['생활인구합계(tot_lvpop_co)'].plot(label = '생활인구합계', xlabel = '헬로', ylabel = '테스트')
plt.legend()
```

![20230516-pp-](https://github.com/jangsukyung/Lecture-Review/assets/130429032/adccb889-a71d-4b16-b7b1-2fdb9f4d6954)

- 그래프 사이즈 조절

```python
plt.figure(figsize = (25, 10))
sample['생활인구합계(tot_lvpop_co)'].plot(label = '생활인구합계')
plt.legend()
```

![20230516-pp-1](https://github.com/jangsukyung/Lecture-Review/assets/130429032/ac4e92a9-8493-4e1b-999c-aa26e43ba8aa)

- 여러개 그래프를 한 번에

```python
figure, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))
```

![20230516-pp-2](https://github.com/jangsukyung/Lecture-Review/assets/130429032/e53e2079-a2e3-4bfa-b6bf-2fad92bc74bc)

```python
figure, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))

sample['생활인구합계(tot_lvpop_co)'].plot(ax = ax1, label = 'total', xlabel = '생활인구합계')
sample['남자0~9세(male_f0t9_lvpop_co)'].plot(ax = ax2, label = 'man 0~9', xlabel = '남자 0~9세')
```

![20230516-pp-3](https://github.com/jangsukyung/Lecture-Review/assets/130429032/38f6206e-27c9-472c-b8a5-d88362fa35e2)