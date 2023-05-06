---
layout: single
title:  "Store Sales TS Forecasting - A Comprehensive Guide"
categories: Transcription
tag: [Python, Kaggle, Store, Sales, Transcription]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

시계열 예측에 대한 포괄적인 가이드

# Store Sales - Time Series Forecasting

Before reading the notebook, what will you learn from this notebook?

- Interpolation for Oil Prices
- Detailed Data Manupulation for Holiday and Events Data
- Exploratory Data Analysis
- Hypothesis Testing
- Modelling

“Store Sales - Time Series Forecasting” 은 상점 매출 데이터를 활용한 시계열 예측에 대한 노트북입니다. 이 노트북에서는 다음과 같은 내용을 배울 수 있습니다.

- 유가 시계열 데이터에 대한 보간법 기술
- 공ㅇ휴일 및 이벤트 데이터의 상세한 데이터 조작 기술
- 탐색적 데이터 분석 방법론
- 가설 검정 기법
- 시계열 모델링 기술

This competition is about time series forcasting for store sales. The data comes from an Ecuador company as known as Corporación Favorita and it is a large grocery retailer. Also, the company operates in other countries in South America.

If you wonder the company, you can click here to learn something about it

이 대회는 상점 매출을 위한 시계열 예측 대회입니다. 데이터는 에콰도르의 대형 식료품 소매업체인 “Corporación Favorita”에서 제공되며, 이 회사는 남미 다른 국가에서도 운영하고 있습니다.

만약 이 회사에 대해 궁금하시다면, 여기를 클릭하여 관련 정보를 알아보실 수 있습니다.

There are 54 stores and 33 product families in the data. The time series starts from 2013-01-01 and finishes in 2017-08-31. However, you know that kaggle gives us splitted two data as train and test. The dates in the test data are for the 15 days after the last date in the training data. Date range in the test data will be very important to us while we are defining a cross-validation strategy and creating new features.

**Our main mission in this competition is, predicting sales for each product family and store combinations.**

이 데이터에는 54개의 상점과 33개의 제품 패밀리가 있습니다. 시계열 데이터는 2013년 1월 1일부터 2017년 8월 31일까지의 기간을 다룹니다. 그러나 Kaggle에선 학습(train)데이터와 테스트(test) 데이터를 분리하여 제공합니다. 테스트 데이터의 날짜는 학습 데이터의 마지막 날짜로부터 15일 이후의 기간에 해당합니다. 테스트 데이터의 날짜 범위는 교차 검증 전략 및 새로운 특징(feature)을 생성하는 과정에서 매우 중요합니다.

**이 대회에서의 주요 임무는 각 제품 패밀리와 상점 조합에 대한 매출을 예측하는 것입니다.**

There are 6 data that we will study on them step by step.

우리는 단계별로 다음 6개의 데이터에 대해 연구할 것입니다.

1. Train (학습 데이터)
2. Test (테스트 데이터)
3. Store (상점 데이터)
4. Transactions (거래 데이터)
5. Holidays and Events (휴일 및 이벤트 데이터)
6. Daily Oil Price (일일 유가 데이터)

**The train data** contains time series of the stores and the product families combination. The sales column gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips). The onpromotion column gives the total number of items in a product family that were being promoted at a store at a given date.

**학습 데이터**는 상점과 제품 패밀리 조합의 시계열 데이터를 포함합니다. 매출 열은 특정 날짜에 상점에서 특정 제품 패밀리에 대한 총 매출을 나타냅니다. 제품이 분수 단위로 판매될 수 있기 때문에(예: 치즈 1.5kg 대신 감자칩 1봉지), 분수 값이 가능합니다. onpromotion열은 상점에서 특정 날짜에 프로모션 대상으로 판매되는 제품 패밀리의 총 수를 나타냅니다.

**Stores data** gives some information about stores such as city, state, type, cluster.

**상점 데이터**는 상점의 도시, 주, 유형, 클러스터 등에 대한 정보를 제공합니다.

**Transaction data** is highly correlated with train’s sales column. You can understand the sales patterns of the stores.

**거래 데이터**는 학습 데이터의 매출 열과 강하게 상관관계가 있습니다. 상점의 매출 패턴을 이해할 수 있습니다.

**Holidays and events data** is a meta data. This data is quite valuable to understand past sales, trend and seasonality components. However, it needs to be arranged. You are going to find a comprehensive data manipulation for this data. That part will be one of the most important chapter in this notebook.

**휴일 및 이벤트 데이터**는 메타 데이터입니다. 이 데이터는 지난 매출, 추세 및 계절성 요소를 이해하는 데 매우 유용하지만 정리가 필요합니다. 이 데이터에 대한 포괄적인 데이터 조작 방법을 제공합니다. 이 부분은 이 노트북에서 가장 중요한 장(chapter) 중 하나입니다.

**Daily Oil Price data** is another data which will help us. Ecuador is an oil-dependent country and it’s economical health is highly vulnerable to shocks in oil prices. That’s why, it will help us to understand which product families affected in positive or negative way by oil price.

**일일 유가 데이터**는 우리를 도와줄 또 다른 데이터입니다. 에콰도르는 석유 의존 국가이며, 유가 충격에 매우 취약합니다. 따라서 유가에 긍정적 또는 부정적으로 영향을 받은 제품 패밀리를 이해하는 데 도움이 될 것입니다.

When you look at the data description, you will see “Additional Notes”. These notes may be significant to catch some patterns or anomalies. I’m sharing them with you to remember.

데이터 설명을 보면 “추가 정보”라는 항목이 있습니다. 이런 메모는 일부 패턴이나 이상 현상을 포착하는 데 중요할 수 있습니다. 이 정보를 기억하기 위해 함께 공유하겠습니다.

- Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
- A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.

- 공공 부문에서 임금은 매월 15일과 마지막 날에 매겨집니다. 이로 인해 슈퍼마켓 매출에 영향을 미칠 수 있습니다.
- 2016년 4월 16일에 에콰도르에 7.8 규모의 지진이 발생했습니다. 사람들은 구호 물품을 기부하는 등의 구호 활동에 참여하여 지진 후 몇 주간 슈퍼마켓 매출에 큰 영향을 미쳤습니다.

# 1. Packages

You can find the packages below what I used.

```python
# BASE
# ---------------------------------------------
import numpy as np
import pandas as pd
import os
import gc
import warnings

# PACF - ACF
# ---------------------------------------------
import statsmodels.api as sm

# DATA VISUALIZATION
# ---------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# CONFIGURATIONS
# ---------------------------------------------
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')
```

- 이 코드는 **파이썬 라이브러리와 설정을 불러오는 코드**입니다.
- numpy: 다차원 배열을 다루는 라이브러리
- pandas: 데이터 분석과 관련된 기능을 제공하는 라이브러리
- os: 운영체제와 상호작용하는 함수를 제공하는 라이브러리
- gc: 파이썬의 가비지 컬렉션 기능을 제어하는 라이브러리
- warnings: 경고 메시지를 제어하는 라이브러리
- statsmodels: 통계 모델링을 제공하는 라이브러리
- matplotlib: 데이터 시각화를 위한 라이브러리
- seaborn: matplotlib을 기반으로 한 데이터 시각화 라이브러리
- plotly: 인터렉티브한 데이터 시각화 라이브러리
- 또한, 설정을 변경하는 코드도 포함되어 있습니다. **pd.set_option() 함수는 pandas에서 출력할 수 있는 최대 열의 개수를 제한하지 않도록 설정**하고, **pd.options.display.float_format은 소수점 아래 두 자리까지 출력**하도록 설정합니다. **warnings.filterwarnings() 함수는 경고 메시지를 출력하지 않도록** 설정합니다.

# 2. Importing Data

```python
# Import
train = pd.read_csv('train.csv', encoding = 'cp949')
test = pd.read_csv('test.csv', encoding = 'cp949')
stores = pd.read_csv('stores.csv', encoding = 'cp949')
sub = pd.read_csv('sample_submission.csv', encoding = 'cp949')
transactions = pd.read_csv('transactions.csv', encoding = 'cp949').sort_values(["store_nbr", "date"])

# Datetime
train["date"] = pd.to_datetime(train.date)
test["date"] = pd.to_datetime(test.date)
transactions["date"] = pd.to_datetime(transactions.date)

# Data types
train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

train.head()
```

![A-Comprehensive-Guide](https://user-images.githubusercontent.com/130429032/235813419-ea3c537b-c588-4d9f-b434-3254d1450d99.png)

- transactions.csv는 store_nbr과 date를 기준으로 정렬되어 불러와집니다.
- 데이터의 datetime 형식은 각각 train과 test, transactions 데이터의 date 열을 to_datetime()함수를 사용하여 datetime 타입으로 변환합니다.
- train 데이터의 onpromotion 열은 float16 형식, sales 열은 float32 형식으로 변환합니다. stores 데이터의 cluster 열은 int8 형식으로 변환합니다.

# 3. Transactions

Let’s start with the transaction data

```python
transactions.head(10)
```

![A-Comprehensive-Guide1](https://user-images.githubusercontent.com/130429032/235813422-573bb4d3-7737-45a9-826d-1ff9b5bcadd9.png)

This feature is highly correlated with sales but first, you are supposed to sum the sales feature to find relationship. Transactions means how many people came to the store or how many invoices created in a day.

이 기능은 매출과 높은 상관관계가 있지만, 관계를 찾기 위해선 먼저 sales feature를 합산해야합니다. Transactions은 하루에 가게를 방문한 사람 수나 생성된 송장(청구서)의 수를 의미합니다.

Sales gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units(1.5kg of cheese, for instance, as opposed to 1 bag of chips).

Sales는 특정 날짜의 특정 가게에서 특정 제품군에 대한 총 매출을 나타냅니다. 제품이 분수 단위로 판매될 수 있기 때문에 (예: 치즈 1.5kg 대신 감자칩 1봉지), 분수 값이 가능합니다.

That’s why, transactions will be one of the relevant features in the model. In the following sections, we will generate new features by using transactions.

그래서 transactions는 모델에서 중요한 기능 중 하나가 될 것입니다. 다음 섹션에서는 transactions를 이용하여 새로운 기능을 생성할 것입니다.

```python
temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how = "left")
print("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(temp.corr("spearman").sales.loc["transactions"]))
px.line(transactions.sort_values(["store_nbr", "date"]), x = 'date', y = 'transactions', color = 'store_nbr', title = "Transactions")
```

![A-Comprehensive-Guide2](https://user-images.githubusercontent.com/130429032/235813424-9375562c-ab10-4640-b1b6-477ab9c94cd5.png)

- ‘train’ 데이터에서 날짜(’date’)와 매장 번호(’store_nbr’)별로 그룹화한 후 해당 그룹에서 판매량(’sales’)을 합산한 결과와 ‘transactions’ 데이터를 병합하여, 날짜와 매장 번호별로 총 판매량과 거래 수(’transactions’)간의 스피어만 상관관계를 출력하고, 거래 수 데이터를 가시화하는 코드입니다.
- 스피어만 상관계수(Spearman Correlation Coefficient)는 변수 간의 상관성을 비선형적으로 계산하는 방식으로, 변수의 상관성을 분석할 때 유용합니다. 위 코드에서는 ‘temp’ 데이터프레임의 스피어만 상관계수를 계산한 후 출력합니다. 그리고 마지막 줄에는 ‘transactions’ 데이터를 매장 번호(’store_nbr’)와 날짜(’date’)순으로 정렬한 후, 매장별 거래 수를 시간별로 가시화합니다.

There is a stable pattern in Transaction. All months are similar except December from 2013 to 2017 by boxplot. In addition, we’ve just seen same pattern for each store in previous plot. Store sales had always increased at the end of the year.

Transaction 데이터에서 안정적인 패턴을 볼 수 있습니다. 2013년부터 2017년까지 12월을 제외한 모든 월이 상자그림에서 유사한 양상을 보입니다. 또한 이전 그래프에서 본 것처럼, 각 매장마다 같은 패턴을 보였습니다. 매장의 매출은 연말에 항상 증가했습니다.

```python
a = transactions.copy()
a["year"] = a.date.dt.year
a["month"] = a.date.dt.month
px.box(a, x = "year", y = "transactions", color = "month", title = "Transactions")
```

![A-Comprehensive-Guide3](https://user-images.githubusercontent.com/130429032/235813426-aa8f6a3d-8136-41a5-9678-d7774e6ecb02.png)

- 이 코드는 transactions 데이터를 복사하여, date 열을 활용하여 year 와 month 열을 새로 만듭니다. 이후, plotly 패키지를 이용하여 boxplot을 그리는데, x축에는 year 값을, y축에는 transactions 값을 지정하고, 색상은 month에 따라 구분하여 그립니다. 이렇게 그려진 boxplot은 연도별 transactions 값을 지정하고, 색상은 month에 따라 구분하여 그립니다.
- 이렇게 그려진 boxplot은 연도별 transactions 값의 분포를 보여주는데, 2013년부터 2017년까지 대부분의 연도에서는 1월부터 11월까지는 비슷한 수준의 transactions 값을 보이지만, 12월에는 상대적으로 높은 값을 보입니다. 이는 연말연시에 매출이 높아지는 경향이 있다는 것을 나타내며 이전 plot에서도 매년 매출이 연말에 증가한다는 것을 확인할 수 있습니다.

Let’s take a look at transactions by using monthly average sales!

We’ve just learned a pattern what increases sales. It was the end of the year. We can see that transactions increase in spring and decrease after spring.

월별 평균 매출을 이용하여 거래 내역을 살펴봅시다!

이전에 매출이 증가하는 패턴을 알았습니다. 그것은 연말이었습니다. 우리는 봄에 거래가 증가하고 봄 이후에 감소한다는 것을 볼 수 있습니다.

```python
a = transactions.set_index("date").resample("M").transactions.mean().reset_index()
a["year"] = a.date.dt.year
px.line(a, x = 'date', y = 'transactions', color = 'year', title = 'Monthly Average Transactions')
```

![A-Comprehensive-Guide4](https://user-images.githubusercontent.com/130429032/235813429-20674efa-ad25-4f8b-b043-83f7f81d17d4.png)

- 해당 코드는 transactions 데이터의 날짜별 거래 수를 월 단위로 평균을 내어 시각화하는 코드입니다. resample 메서드를 사용하여 월 단위의 거래 수의 평균 값을 계산하고, 이를 선 그래프로 나타내어 각 연도별로 월별 평균 거래 수의 변화를 살펴볼 수 있습니다.
- 이 그래프는 월별 평균 거래량을 나타내는 것입니다. 봄철에 거래량이 증가하고 이후에는 감소하는 것을 볼 수 있습니다. 또한 매년 말에 매우 높은 거래량을 보이는 것으로 나타났습니다. 이전 그래프에서 본 것과 같은 상점 판매 증가 패턴을 볼 수 있습니다.

When we look at their relationship , we can see that there is a highly correlation between total sales and transactions also

전체 판매액과 거래 횟수 간의 상관 관계가 높은 것으로 나타납니다.

```python
px.scatter(temp, x = "transactions", y = "sales", trendline = "ols", trendline_color_override = "red")
```

![A-Comprehensive-Guide5](https://user-images.githubusercontent.com/130429032/235813431-0ccc9773-582b-48dc-a1a4-13a8cd4b7933.png)

- 이 코드는 산점도와 함께 ols(Ordinary Least Squares) 추세선을 보여주는 Plotly Express의 scatter 함수를 이용합니다.
- x축은 transactions, y축은 sales 값을 가지며, 각 데이터 포인트들은 산점도 형태로 나타나게 됩니다. 또한 trendline 인자를 이용해 추세선을 추가할 수 있습니다. 이 코드에서는 ols 모델을 이용해 추세선을 생성하고, trendline_color_override 인자를 이용해 추세선의 색상을 빨간색으로 설정했습니다.
- 코드 실행 결과, transactions와 sales 간의 강한 양의 상관관계가 있음을 확인할 수 있습니다. 추세선이 상승하는 방향으로 그려져 있으므로, transactions가 높아질수록 sales도 높아지는 경향을 보인다는 것을 알 수 있습니다.

The days of week is very important for shopping. It shows us a great pattern. Stores make more transactions at weekends. Almost the patterns are same from 2013 to 2017 and Saturday is the most important day for shopping.

요일은 쇼핑에 매우 중요한 역할을 합니다. 주말에 상점들은 더 많은 거래를 합니다. 거의 모든 해에 걸쳐 패턴은 같으며, 토요일이 가장 중요한 쇼핑일입니다.

```python
a = transactions.copy()
a["year"] = a.date.dt.year
a["dayofweek"] = a.date.dt.dayofweek + 1
a = a.groupby(["year", "dayofweek"]).transactions.mean().reset_index()
px.line(a, x = "dayofweek", y = "Transactions", color = "year", title = "transactions")
```

![A-Comprehensive-Guide6](https://user-images.githubusercontent.com/130429032/235813435-79452e4e-2f0f-48ed-824b-32e4a48b8eac.png)

- 위 코드는 transactions 데이터셋을 활용하여 매년 요일별 평균 거래량을 시각화하는 코드입니다.
- 우선 transactions 데이터셋의 날짜 정보를 기반으로 년도(year)와 요일(dayofweek) 정보를 생성합니다. 이후 year과 dayofweek를 그룹화하여 각 그룹의 거래량 평균을 구합니다. 이렇게 구한 거래량 평균을 요일(dayofweek)에 대해 선 그래프로 시각화합니다.
- 시각화 결과, 요일(dayofweek)이 거래량에 미치는 영향이 크다는 것을 알 수 있습니다. 주말에 거래량이 증가하는 경향이 있으며, 특히 토요일의 거래량이 가장 높은 것으로 나타났습니다. 또한 2013년부터 2017년까지의 거래량 패턴이 대체로 유사하다는 것을 알 수 있습니다.

# 4. Oil Price

The economy is one of the biggest problem for the governments and people. It affects all of things in a good or bad way. In our case, Ecuador is an oil-dependent country. Changing oil prices in Ecuador will cause a variance in the model. I researched Ecuador’s economy to be able to understand much better and I found an article from IMF. You are supposed to read it if you want to make better models by using oil data.

경제는 정부와 국민들에게 가장 큰 문제 중 하나입니다. 좋은 방법으로 또는 나쁜 방법으로 모든 것에 영향을 미칩니다. 우리 경우, 에콰도르는 석유 의존적인 나라입니다. 에콰도르의 석유 가격 변동은 모델에 차이를 일으킬 수 있습니다. 에콰도르의 경제에 대해 조사하여 더 잘 이해하기 위해 IMF에서 논문을 찾았습니다. 석유 데이터를 사용하여 더 나는 모델을 만들고 싶다면 이 논문을 읽어보시기를 권장합니다.

There are some missing data points in the daily oil data as you can see below. You can treat the data by using various imputation methods. However, I choose a simple solution for that. Linear Interpolation is suitable for this time series. You can see the trend and predict missing data points, when you look at a time series plot of oil price.

아래에서 볼 수 있듯이 일일 유가 데이터에는 일부 결측치가 있습니다. 이를 해결하기 위해 다양한 보간 방법을 사용할 수 있지만, 여기서는 간단한 해결책으로 선형 보간법을 선택했습니다. 유가 가격의 시계열 그래프를 보면 추세를 파악하고 결측치를 예측할 수 있습니다.

```python
# Import
oil = pd.read_csv("oil.csv", encoding = 'cp949')
oil["date"] = pd.to_datetime(oil.date)

# Resample
oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()

# Interpolate
oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()

# Plot
p = oil.melt(id_vars = ['date'] + list(oil.keys()[5:]), var_name = 'Legend')
px.line(p.sort_values(["Legend", "date"], ascending = [False, True]), x = 'date', y = 'value', color = 'Legend', title = "Daily Oil Price")
```

![A-Comprehensive-Guide7](https://user-images.githubusercontent.com/130429032/235813437-ae69dac5-5e10-4bc8-868d-0bfc773fd68f.png)

- 이 코드는 원유의 일일 가격을 나타내는 시계열 데이터셋 oil.csv 를 불러와 “date” 컬럼을 datetime 형식으로 변환합니다.
- “date”컬럼을 인덱스로 설정하고 “dcoilwtico” 컬럼을 일일 주파수로 재샘플링합니다. 그리고 다시 인덱스를 리셋합니다.
- “dcoilwtico”컬럼에서 값이 0인 경우에는 NaN으로 바꾸어줍니다.
- 보간된 값을 저장할 새로운 “dcoilwtico_interpolated” 컬럼을 만듭니다.
- “id_vars”에는 “date”컬럼을, “var_name”에는 “Legend”변수명을 할당하고, “value_vars”에는 “dcoilwtico”와 “dcoilwtico_interpolated” 컬럼을 제외한 나머지 컬럼들을 할당합니다.
- “Legend”와 “date”를 기준으로 데이터를 오름차순 정렬하고 px.line()을 사용하여 시각화힙니다.

I just said, “Ecuador is a oil-dependent country” but is it true? Can we really see that from the data by looking at?

저는 “에콰도르는 석유 의존국이다”라고 말했는데, 이게 사실인지 데이터를 통해 확인할 수 있을까요?

First of all, let’s look at the correlations for sales and transactions. The correlation values are not strong but the sign of sales is negative. Maybe, we can catch a clue. Logically, if daily oil price is high, we expect that the Ecuador’s economy is bad and it means the price of product increases and sales decreases. There is a negative relationship here.

우선, 매출과 거래량 간의 상관 관계를 살펴보겠습니다. 상관 관계 값은 강하지 않지만 매출의 부호가 음수입니다. 아마도 우리는 단서를 잡을 수 있을 것입니다. 논리적으로 말하자면, 일일 유가가 높다면 에콰도르의 경제가 나쁠 것으로 예상되며, 제품 가격이 오르고 매출이 감소하는 것을 의미합니다. 여기에는 부정적인 관계가 있습니다.

```python
temp = pd.merge(temp, oil, how = "left")
print("Correlation with Daily Oil Prices")
print(temp.drop(["store_nbr", "dcoilwtico"], axis = 1).corr("spearman").dcoilwtico_interpolated.loc[["sales", "transactions"]], "\n")

fig, axes = plt.subplots(1, 2, figsize = (15, 5))
temp.plot.scatter(x = "dcoilwtico_interpolated", y = "transactions", ax = axes[0])
temp.plot.scatter(x = "dcoilwtico_interpolated", y = sales", ax = axes[1], color = "r")
axes[0].set_title('Daily Oil Price & Transactions', fontsize = 15)
axes[1].set_title('Daily Oil Price & Sales', fontsize = 15)
```

![A-Comprehensive-Guide8](https://user-images.githubusercontent.com/130429032/235813438-c448e991-c2c9-4ca4-b360-bb34559d735a.png)

- 위 코드는 먼저 ‘oil’ 데이터를 ‘temp’ 데이터에 병합하고, ‘sales’와 ‘transactions’과의 스피어만 순위 상관 관계를 계산하여 출력합니다. 그리고 이어서 daily oil price와 ‘transactions’, ‘sales’ 간의 산점도 그래프를 그립니다.
- ‘axes[0]’은 daily oil price와 ‘transactions’간의 산점도 그래프를, ‘axes[1]’은 daily oil price와 ‘sales’간의 산점도 그래프를 나타내며, 빨간색으로 표시됩니다. 각각의 그래프에는 각 축의 레이블과 그래프 제목이 있습니다. 이를 통해, daily oil price와 ‘sales’, ‘transactions’ 간의 관계를 시각적으로 알 수 있습니다.

You should never decide what you will do by looking at a graph or result! You are supposed to change your view and define new hypotheses.

We would have been wrong if we had looked at some simple outputs just like above and we had said that there is no relationship with oil prices and let’s not use oil price data.

All right! We are aware of anlyzing deeply now. Let’s draw a scatter plot but let’s pay attention for product families this time. All of the plots almost contains same pattern. When daily oil price is under about 70, there are more sales in the data. There are 2 cluster here. They are over 70 and under 70. It seems pretty understandable actually.

We are in a good way I think. What do you think? Just now, we couldn’t see a pattern for daily oil price, but now we extracted a new pattern from it.

그러나 결과물을 보고서 결정을 내려선 안됩니다. 대신, 새로운 가설을 정의하고 그에 따라 분석을 해야 합니다.

만약 위와 같은 간단한 결과를 보고서 석유 가격과 매출 간의 상관 관계가 없다고 결론지었다면 우리는 잘못됐을 것입니다.

맞습니다. 이제 우리는 깊이 있는 분석 방법을 알고 있습니다. 제품군에 따라 산점도를 그리되 이번에는 일일 유가에 주목해 봅시다. 대부분의 그래프가 거의 동일한 패턴을 보입니다. 일일 유가가 약 70달러 미만일 때 데이터에서 더 많은 매출이 발생합니다. 여기에는 2개의 군집이 있습니다. 70달러를 초과하거나 미만인 것입니다. 꽤 이해하기 쉬운 패턴입니다.

우리는 좋은 길을 가고 있다고 생각합니다. 당신은 어떻게 생각하시나요? 방금까지 일일 유가에 대한 패턴을 볼 수 없었지만, 이제 그 중에서 새로운 패턴을 추출해냈습니다.

```python
a = pd.merge(train.groupby(["date", "family"]).sales.sum().reset_index(), oil.drop("dcoilwtico", axis = 1), how = "left")
c = a.groupby("family").corr("spearman").reset_index()
c = c[c.level_1 == "dcoilwtico_interpolated"][["family", "sales"]].sort_values("sales")

fig, axes = plt.subplots(7, 5, figsize = (20,20))
for i, fam in enumerate(c.family):
    if i < 6:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[0, i-1])
        axes[0, i-1].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[0, i-1].axvline(x=70, color='r', linestyle='--')
    if i >= 6 and i<11:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[1, i-6])
        axes[1, i-6].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[1, i-6].axvline(x=70, color='r', linestyle='--')
    if i >= 11 and i<16:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[2, i-11])
        axes[2, i-11].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[2, i-11].axvline(x=70, color='r', linestyle='--')
    if i >= 16 and i<21:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[3, i-16])
        axes[3, i-16].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[3, i-16].axvline(x=70, color='r', linestyle='--')
    if i >= 21 and i<26:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[4, i-21])
        axes[4, i-21].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[4, i-21].axvline(x=70, color='r', linestyle='--')
    if i >= 26 and i < 31:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[5, i-26])
        axes[5, i-26].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[5, i-26].axvline(x=70, color='r', linestyle='--')
    if i >= 31 :
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[6, i-31])
        axes[6, i-31].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[6, i-31].axvline(x=70, color='r', linestyle='--')
        
        
plt.tight_layout(pad=5)
plt.suptitle("Daily Oil Product & Total Family Sales \n", fontsize = 20);
plt.show()
```

![A-Comprehensive-Guide9](https://user-images.githubusercontent.com/130429032/235813444-a9221140-8302-47b9-bed5-eb7ac8d2727c.png)

# 5. Sales

Our main objective is, predicting store sales for each product family. For this reason, sales column should be examined more seriously. We need to learn everything such as seasonality, trends, anomalies, similarities with other time series and so on.

우리의 주요 목표는 각 제품군의 매장 매출을 예측하는 것입니다. 이를 위해서 매출 열을 계절성, 추세, 이상치, 다른 시계열과의 유사성 등 모든 면에서 더 심층적으로 분석해야 합니다.

Most of the stores are similar to each other, when we examine them with correlation matrix. Some stores, such as 20, 21, 22, and 52 may be a little different.

상권간의 상관관계 행렬을 사용하여 대부분의 상점들이 서로 유사하다는 것을 알 수 있습니다. 그러나 20, 21, 22, 52와 같은 일부 상점은 다른 상점과는 조금 다를 수 있습니다.

```python
a = train[["store_nbr", "sales"]]
a["ind"] = 1
a["ind"] = a.groupby("store_nbr").ind.cumsum().values
a = pd.pivot(a, index = "ind", columns = "store_nbr", values = "sales").corr()
mask = np.triu(a.corr())
plt.figure(figsize = (20, 20))
sns.heatmap(a,
					annot = True,
					fmt = '.1f',
					cmap = 'coolwarm',
					square = True,
					mask = mask,
					linewidths = 1,
					cbar = False)
plt.title("Correlations among stores", fontsize = 20)
plt.show()
```

![A-Comprehensive-Guide10](https://user-images.githubusercontent.com/130429032/235813449-be66d15e-a6c3-499d-98f4-01b14df95d37.png)

- 위 코드는 train 데이터프레임에서 “store_nbr”과 “sales”컬럼만 추출하여 이들 간의 상관관계를 분석하는 코드입니다.
- 먼저 “a”라는 변수에 “store_nbr”과 “sales” 컬럼을 저장하고, 각 상점마다 구분을 위한 “ind”컬럼을 추가합니다. “ind”컬럼은 상점 번호마다 1부터 시작되는 일련번호를 부여하는데, 이는 상점별 매출 데이터를 행 방향으로 정리하기 위해 사용됩니다.
- 그리고 나서, “a”데이터프레임을 pivot table 형태로 변환하고, 각 상점별 매출 데이터를 행으로 정리합니다. 그리고 이를 기반으로 상점 간의 상관관계를 계산합니다.
- 마지막으로, 계산된 상관관계를 시각화하는데, 여기선 seaborn 패키지의 heatmap 함수를 사용하여 열린 삼각형 영역에만 상관계수 값을 표시하고, 그 외 삼각형 영역은 마스크 처리하여 시각적으로 표현합니다. 이를 통해 각 상점간의 상관관계를 한 눈에 파악할 수 있습니다.

There is a graph that shows us daily total sales below.

일일 총 매출을 보여주는 그래프가 있습니다.

```python
a = train.set_index("date").groupby("store_nbr").resample("D").sales.sum().reset_index()
px.line(a, x = "date", y = "sales", color = "store_nbr", title = "Daily total sales of the stores")
```

![A-Comprehensive-Guide11](https://user-images.githubusercontent.com/130429032/235813452-d8ab68bd-7485-4e32-8a4d-b6bbf0bfbd44.png)

- 위 코드는 train 데이터에서 날짜(date)를 기준으로 매일(store_nbr별) 총 매출(sales)을 합산한 데이터를 생성하고, 이를 가지고 각 매장(store_nbr)의 일일 총 매출을 선 그래프로 시각화하는 코드입니다.

I realized some unnecessary rows in the data while I was looking at the time series of the stores one by one, If you select the stores from above, some of them have no sales at the beginning of 2013. You can see them, if you look at the those stores 20, 21, 22, 29, 36, 42, 52 and 53. I decided to remove those rows before the stores opened. In the following codes, we will get rid of them.

일련의 매장을 하나씩 살펴보면서 데이터에서 불필요한 행들을 발견했습니다. 위에서 선택한 일부 매장들은 2013년 초에는 매출이 없는 것으로 나타났습니다. 이는 20, 21, 22, 29, 36, 42, 52, 53번 매장들에서 확인할 수 있습니다. 따라서 이러한 매장들이 영업을 시작하기 전의 행들을 제거하기로 결정했습니다. 다음 코드에서 이를 처리하겠습니다.

```python
print(train.shape)
train = train[~((train.store_nbr == 52) * (train.date < "2017-04-20"))]
train = train[~((train.store_nbr == 22) * (train.date < "2015-10-09"))]
train = train[~((train.store_nbr == 42) * (train.date < "2015-08-21"))]
train = train[~((train.store_nbr == 21) * (train.date < "2015-07-24"))]
train = train[~((train.store_nbr == 29) * (train.date < "2015-03-20"))]
train = train[~((train.store_nbr == 20) * (train.date < "2015-02-13"))]
train = train[~((train.store_nbr == 53) * (train.date < "2014-05-29"))]
train = train[~((train.store_nbr == 36) * (train.date < "2013-05-09"))]
train.shape

# 출력값
# (3000888, 6)
# (2780316, 6)
```

## Zero Forecasting

Some stores don’t sell some product families. In the following code, you can see which products aren’t sold in which stores. It isn’t difficult to forecast them next 15 days. Their forecasts must be 0 next 15 days.

일부 매장들은 일부 제품군을 판매하지 않습니다. 다음 코드에서 어떤 제품군이 어떤 매장에서 판매되지 않는지 볼 수 있습니다. 이들은 다음 15일 동안 예측할 때 예측값이 0이 되어야 합니다.

I will remove them from the data and create a new data frame for product families which never sell. Then, when we are at submission part, I will combine that data frame with our predictions.

이들을 데이터에서 제거하고, 결측값이 없는 제품군에 대해 새로운 데이터 프레임을 만들 것입니다. 그런 다음 제출 부분에서 그 데이터 프레임을 우리의 예측과 결합할 것입니다.

```python
c = train.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_Values(["family", "store_nbr"])
c = c[c.sales == 0]
c
```

- 이 코드는 store_nbr과 family를 그룹화하여 sales 항목의 합계를 구하고, 그 합계가 0인 경우를 필터링합니다. 이렇게 필터링된 데이터프레임을 변수 c에 할당합니다. 이 데이터프레임은 해당 가게에서 해당 제품군이 판매되지 않는 경우를 보여줍니다.

```python
print(train.shape)
# Anti Join
outer_join = train.merge(c[c.sales == 0].drop("sales", axis = 1), how = 'outer', indicator = True)
train = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis = 1)
del outer_join
gc.collect()
train.shape

# 출력값
# (2780316, 6)
# (2698648, 6)
```

- 위 코드는 데이터셋에서 제품군 중 판매량이 0인 제품들과 연관된 매장을 찾아서, 해당 매장에서는 해당 제품군을 예측할 필요가 없다는 것을 확인하고, 이를 데이터셋에서 제거하는 코드입니다.
- 매장별, 제품군별 총 판매량을 계산하고, 판매량이 0인 데이터만 추려서 c 데이터프레임에 저장합니다.
- 그 다음, ‘train’ 데이터셋과 ‘c’데이터 프레임을 ‘store_nbr’, ‘family’ 기준으로 outer join 하여, 판매량이 0인 제품과 연관된 매장 데이터를 가져옵니다. ‘indicator’ 옵션을 사용하여 join의 결과를 ‘_merge’컬럼에 저장합니다.
- 마지막으로 ‘_merge’컬럼에서 ‘both’인 데이터를 제외한 나머지 데이터만 ‘train’데이터 셋으로 가져옵니다. 이를 통해 판매량이 0인 제품과 연관된 매장 데이터를 제거합니다. 그리고 메모리 절약을 위해 ‘outer_join’ 데이터 프레임을 삭제하고, 불필요한 메모리를 제거합니다.

```python
zero_prediction = []
for i in range(0, len(c)):
	zero_prediction.append(
			pd.DataFrame({
					"date":pd.date_range("2017-08-16", "2017-08-31").tolist(),
					"store_nbr":c.store_nbr.iloc[i],
					"family":c.family.iloc[i],
					"sales":0
			})
	)
zero_prediction = pd.concat(zero_prediction)
del c
gc.collect()
zero_prediction
```

![A-Comprehensive-Guide12](https://user-images.githubusercontent.com/130429032/235813453-5ebb5585-fc70-413c-9eec-25c630be832e.png)

- 위 코드는 이전에 구한 ‘c’ 데이터프레임에서 ‘sales’값이 0인 제품군을 선택하여 이후 15일간 예측값을 0으로 설정하는 작업을 수행하는 코드입니다.
- 2017년 8월 17일부터 8월 31일까지의 날짜를 리스트 형태로 생성하고, 해당 기간 내에서 제품군, 매장, 매출액이 모두 0인 데이터 프레임을 생성합니다. 이 작업을 제품군의 개수만큼 반복하여 예측값이 0인 데이터프레임을 생성하고 이를 하나의 데이터프레임으로 합칩니다.
- 이후 ‘del c’와 ‘gc.collect()’를 통해 불필요한 데이터를 삭제하고 메모리를 최적화합니다.

Our first forecasting is done! You don’t need the machine learning or deep learning or another things for these time series because we had some simple time series

첫 번째 예측이 완료되었습니다! 이러한 시계열 데이터에선 머신러닝, 딥러닝 또는 다른 복잡한 방법들이 필요하지 않습니다.

## Are the product Families Active or Passive?

제품군은 능동적입니까 아니면 수동적입니까?

some products can sell rarely in the stores. When I worked on a product supply demand for restuarants project at my previous job, some products were passive if they never bought in the last two months. I want to apply this domain knowledge here and I will look on the last 60 days.

제품군 중 일부는 가게에서 드물게 팔릴 수 있습니다. 이전 직장에서 레스토랑 제품 공급 수요 프로젝트를 작업할 때, 지난 두 달 동안 구매되지 않은 제품은 패시브 제품으로 분류했습니다. 이러한 도메인 지식을 여기에 적용하고, 마지막 60일 동안 판매가 없는 제품군은 ‘패시브’ 상태로 간주하겠습니다.

However, some product families depends on seasonality. Some of them might not active on the last 60 days but it doesn’t mean it is passive.

그러나 일부 제품군은 계절에 따라 판매 수요가 변동할 수 있습니다. 마지막 60일 동안 활성화되지 않았더라도 패시브 상태라고 단정짓지 않을 필요가 있습니다.

```python
c = train.groupby(["family", "store_nbr"]).tail(60).groupby(["family", "store_nbr"]).sales.sum().reset_index()
c[c.sales == 0]
```

As you can see below, these examples are too rare and also the sales are low. I’m open your suggestions for these families. I won’t do anything for now but, you would like to improve your model you can focus on that.

아래에서 보듯이, 이런 예시는 너무 드문 경우이며, 판매량도 낮습니다. 이러한 패밀리에 대해 아이디어가 있으면 제안해주시며 감사하겠습니다. 일단은 그대로 두지만, 모델을 개선하려면 이에 초점을 맞출 수 있습니다.

But still, I want to use that knowledge whether it is simple and I will create a new feature. It shows that the product family is active or not.

하지만 여전히 그 지식을 활용하고자 하며, 간단하지만 새로운 feature를 생성하여 해당 제품 패밀리가 활성화되어 있는지 여부를 나타내고자 합니다.

```python
fig, ax = plt.subplots(1, 5, figsize = (20, 4))
train[(train.store_nbr == 10) & (train.family == "LAWN AND GARDEN")].set_index("date").sales.plot(Ax = ax[0], title = "STORE 10 - LAWN AND GARDEN")
train[(train.store_nbr == 36) & (train.family == "LADIESWEAR")].set_index("date").sales.plot(ax = ax[1], title = "STORE 36 - LADIESWEAR")
train[(train.store_nbr == 6) & (train.family == "SCHOOL AND OFFICE SUPPLIES")].set_index("date").sales.plot(ax = ax[2], title = "STORE 6 - SCHOOL AND OFFICE SUPPLIES")
train[(train.store_nbr == 14) & (train.family == "BABY CARE")].set_index("date").sales.plot(ax = ax[3], title = "STORE 14 - BABY CARE")
train[(train.store_nbr == 53) & (train.family == "BOOKS")].set_index("date").sales.plot(ax = ax[4], title = "STORE 43 - BOOKS")
plt.show()
```

![A-Comprehensive-Guide13](https://user-images.githubusercontent.com/130429032/235813456-5c1e88ae-5928-48b6-8172-f85fdca816c8.png)

We can catch the trends, seasonality and anomalies for families.

제품군 단위로 판매를 분석하면 추세, 계절성 및 이상치 등을 파악할 수 있습니다.

```python
a = train.set_index("date").groupby("family").resample("D").sales.sum().reset_index()
px.line(a, x = "date", y = "sales", color = "family", title = "Daily total sales of the family")
```

![A-Comprehensive-Guide14](https://user-images.githubusercontent.com/130429032/235813459-bde901e3-4a04-467f-98af-6d7d6b2001df.png)

We are working with the stores. Well, there are plenty of products in the stores and we need to know which product family sells much more? Let’s make a barplot to see that.

가게에서 다양한 제품들이 판매되고 있기 때문에 어떤 제품군이 가장 많이 판매되는지 알아야 합니다. 이를 확인하기 위해 막대 그래프를 그려보겠습니다.

The graph shows us GROCERY I and  BEVERAGES are the top selling families.

그래프는 GROCERY I와 BEVERAGES가 가장 많이 판매되는 제품군임을 보여줍니다.

```python
a = train.groupby("family").sales.mean().sort_values(ascending = False).reset_index()
px.bar(a, y = "family", x = "sales", color = "family", title = "Which product family preferred more?")
```

![A-Comprehensive-Guide15](https://user-images.githubusercontent.com/130429032/235813463-fcea6e86-b731-4d69-b9c6-d716abb87fc5.png)

Does onpromotion column cause a data leakage problem?

“onpromotion”열은 데이터 누수 문제를 일으킬까요?

It is really a good question. The Data Leakage is one of the biggest problem when we will fit a model. There is a great discussion from Nesterenko Marina @nesterenkomarina. You should look at it before fitting a model.

실제로 좋은 질문입니다. 데이터 누수(Data Leakage)는 모델을 학습시킬 때 가장 큰 문제 중 하나입니다. Nesterenko Marina @nesterenkomarina에 좋은 논의가 있으니 모델을 적합하기 전에 참고하시는 것이 좋습니다.

```python
print("Spearman Correlation between Sales and Onpromotion: {:,.4f}".format(train.corr("spearman")sales.loc["onpromotion"]))
# 출력값
# Spearman Correlation between Sales and Onpromotion: 0.5304
```

How different can stores be from each other? I couldn’t find a major pattern among the stores actually. But I only looked at a single plot. There may be some latent patterns.

가게 간에 어떤 차이가 있을까요? 실제로 주요한 패턴을 찾지 못했습니다. 하지만 하나의 그래프만 봤을 뿐입니다. 잠재적인 패턴이 있을 수 있습니다.

```python
d = pd.merge(train, stores)
d["store_nbr"] = d["store_nbr"].astype("int8")
d["year"] = d.date.dt.year
px.line(d.groupby(["city", "year"]).sales.mean().reset_index(), x = "year", y = "sales", color = "city")
```

![A-Comprehensive-Guide16](https://user-images.githubusercontent.com/130429032/235813467-d5f7fccd-3557-49ba-ae3a-9b479c1091a5.png)

# 6. Holidays and Events

What a mess! Probably, you are confused due to the holidays and events data. It contains a lot of information inside but, don’t worry. You just need to take a breathe and think! It is a meta-data so you have to split it logically and make the data useful.

아마도 휴일 및 이벤트 데이터 때분에 혼란스러울 것입니다. 그 안에는 많은 정보가 들어있지만 걱정하지 마세요. 먼저 깊게 한 번 생각해 보세요! 이것은 메타데이터이므로 논리적으로 분할하고 데이터를 유용하게 만들어야 합니다.

What are our potblems?

- Some national holidays have been transferred.
- There might be a few holidays in one day. When we merged all of data, number of rows might increase. We don’t want duplicates
- What is the scope of holidays? It can be regional or national or local. You need to split them by the scope.
- Work day issue
- Some specific events
- Creating new features etc.

End of the section, they won’t be a problem anymore!

우리의 문제는 무엇인가요?

- 일부 국경일이 이전되었습니다.
- 하루에 여러 개의 휴일이 있을 수 있습니다. 모든 데이터를 병합하면 행 수가 증가할 수 있습니다. 우리는 중복을 원하지 않습니다.
- 휴일의 범위는 무엇인가요? 종교적, 지역적, 국가적일 수 있습니다. 범위별로 분리해야합니다.
- 근무일 문제
- 특정 이벤트
- 새로운 기능 생성 등

이 섹션이 끝나면 더 이상 문제가 되지 않을 것입니다!

```python
holidays = pd.read_csv("holidays_events.csv", encoding = "cp949")
holidays["date"] = pd.to_datetime(holidays.date)

# holidays[holidays.type == "Holiday"]
# holidays[(holidays.type == "Holiday") & (holidays.transferred == True)]

# Transferred Holidays
tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis = 1).reset_index(drop = True)
tr2 = holidays[(holidays.type == "Transfer")].drop("transferred", axis = 1).reset_index(drop = True)
tr = pd.concat([tr1,tr2], axis = 1)
tr = tr.iloc[:, [5,1,2,3,4]]

holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis = 1)
holidays = holidays.append(tr).reset_index(drop = True)

# Additional Holidays
holidays["description"] = holidays["description"].str.replace("-", "").str.replace("+", "").str.replace('\d+', '')
holidays["type"] = np.where(holidays["type"] == "Additional", "Holiday", holidays["type"])

# Bridge Holidays
holidays["description"] = holidays["description"].str.replace("Puente ", "")
holidays["type"] = np.where(holidays["type"] == "Bridge", "Holiday", holidays["type"])

 
# Work Day Holidays, that is meant to payback the Bridge.
work_day = holidays[holidays.type == "Work Day"]  
holidays = holidays[holidays.type != "Work Day"]  

# Split

# Events are national
events = holidays[holidays.type == "Event"].drop(["type", "locale", "locale_name"], axis = 1).rename({"description":"events"}, axis = 1)

holidays = holidays[holidays.type != "Event"].drop("type", axis = 1)
regional = holidays[holidays.locale == "Regional"].rename({"locale_name":"state", "description":"holiday_regional"}, axis = 1).drop("locale", axis = 1).drop_duplicates()
national = holidays[holidays.locale == "National"].rename({"description":"holiday_national"}, axis = 1).drop(["locale", "locale_name"], axis = 1).drop_duplicates()
local = holidays[holidays.locale == "Local"].rename({"description":"holiday_local", "locale_name":"city"}, axis = 1).drop("locale", axis = 1).drop_duplicates()

d = pd.merge(train.append(test), stores)
d["store_nbr"] = d["store_nbr"].astype("int8")

# National Holidays & Events
#d = pd.merge(d, events, how = "left")
d = pd.merge(d, national, how = "left")
# Regional
d = pd.merge(d, regional, how = "left", on = ["date", "state"])
# Local
d = pd.merge(d, local, how = "left", on = ["date", "city"])

# Work Day: It will be removed when real work day colum created
d = pd.merge(d,  work_day[["date", "type"]].rename({"type":"IsWorkDay"}, axis = 1),how = "left")

# EVENTS
events["events"] =np.where(events.events.str.contains("futbol"), "Futbol", events.events)

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
    # categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    df.columns = df.columns.str.replace(" ", "_")
    return df, df.columns.tolist()

events, events_cat = one_hot_encoder(events, nan_as_category=False)
events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1,events["events_Dia_de_la_Madre"])
events = events.drop(239)

d = pd.merge(d, events, how = "left")
d[events_cat] = d[events_cat].fillna(0)

# New features
d["holiday_national_binary"] = np.where(d.holiday_national.notnull(), 1, 0)
d["holiday_local_binary"] = np.where(d.holiday_local.notnull(), 1, 0)
d["holiday_regional_binary"] = np.where(d.holiday_regional.notnull(), 1, 0)

# 
d["national_independence"] = np.where(d.holiday_national.isin(['Batalla de Pichincha',  'Independencia de Cuenca', 'Independencia de Guayaquil', 'Independencia de Guayaquil', 'Primer Grito de Independencia']), 1, 0)
d["local_cantonizacio"] = np.where(d.holiday_local.str.contains("Cantonizacio"), 1, 0)
d["local_fundacion"] = np.where(d.holiday_local.str.contains("Fundacion"), 1, 0)
d["local_independencia"] = np.where(d.holiday_local.str.contains("Independencia"), 1, 0)

holidays, holidays_cat = one_hot_encoder(d[["holiday_national","holiday_regional","holiday_local"]], nan_as_category=False)
d = pd.concat([d.drop(["holiday_national","holiday_regional","holiday_local"], axis = 1),holidays], axis = 1)

he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist()+ d.columns[d.columns.str.startswith("local")].tolist()
d[he_cols] = d[he_cols].astype("int8")

d[["family", "city", "state", "type"]] = d[["family", "city", "state", "type"]].astype("category")

del holidays, holidays_cat, work_day, local, regional, national, events, events_cat, tr, tr1, tr2, he_cols
gc.collect()

d.head(10)
```

Let’s apply an AB test to Events and Holidays features. Are they Statistically significant? Also it can be a good way for first feature selection.

이벤트와 휴일 기능에 AB 테스트를 적용해보겠습니다. 이들이 통계적으로 유의한가요? 또한 첫번째 특성 선택에 좋은 방법일 수 있습니다.

H0: The sales are equal (M1 = M2)

H1: The sales are not equal(M1 ≠ M2)

H0: 매출이 동일합니다. (M1 = M2)

H1: 매출이 동일하지 않습니다. (M1 ≠ M2)

```python
def AB_Test(dataframe, group, target):

    # Packages
    from scipy.stats import shapiro
    import scipy.stats as stats

    # Split A/B
    groupA = dataframe[dataframe[group] == 1][target]
    groupB = dataframe[dataframe[group] == 0][target]

    # Assumption : Normality
    ntA = shapiro(groupA)[1] < 0.05
    ntB = shapiro(groupB)[1] < 0.05
    # H0 : Distribution is Normal! - False
    # H1 : Distribution is not Normal! - True

    if (ntA == False) & (ntB == False): # "H0: Normal Distribution"

        # Parametric Test
        # Assumption: Homogeneity of variances
        leveneTest = stats.levene(groupA, groupB)[1] < 0.05
        # H0: Homogeneity: False
        # H1: Heterogeneous: True
        
        if leveneTest == False:
            # Homogeneity
            ttest = stats.ttest_ind(groupA, groupB, equal_var = True)[1]

            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
        else:
            # Hetergeneous
            ttest = stats.ttest_ind(groupA, groupB, equal_var = False)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
    else:
        # Non-parametri Test
        ttest = stats.mannwhitneyu(groupA, groupB)[1]
        # H0: M1 == M2 - False
        # H1: M1 != M2 - True

    # Result
    temp = pd.DataFrame({
        "AB Hypothesis": [ttest < 0.05],
        "p-value": [ttest]
    })
    temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
    temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
    temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar!", "A/B groups are not similar!")
    temp["Feature"] = group
    temp["GroupA_mean"] = groupA.mean()
    temp["GroupB_mean"] = groupB.mean()
    temp["GroupA_median"] = groupA.median()
    temp["GroupB_median"] = groupB.median()

    # Columns
    if (ntA == False) & (ntB == False):
        temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
        temp = temp[["Feature", "Test Type", "Homogeneity", "AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
    else:
        temp = temp[["Feature", "Test Type", "AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]

    # Print Hypothesis
    # print("# A/B Testing Hypothesis")
    # print("H0: A == B")
    # print("H1: A != B", "\n")

    return temp

# Apply A/B Testing
he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist() + d.columns[d.columns.str.startswith("local")].tolist()
ab = []
for i in he_cols:
    ab.append(AB_Test(dataframe = d[d.sales.notnull()], group = i, target = "sales"))
ab = pd.concat(ab)
ab
```

- 이 코드는 A/B 테스트를 수행하는 함수를 구현하고, 이 함수를 이용해 여러 feature들에 대한 A/B 테스트 결과를 출력하는 코드입니다.
- AB_Test함수는 데이터프레임, 그룹 변수, 대상 변수를 입력 받습니다. 먼저 입력된 그룹 변수를 이용해 A 그룹과 B 그룹으로 데이터를 분리합니다. 그 다음, 입력된 대상 변수의 분포가 정규분포를 따르는지 검정합니다. 만약 양쪽 그룹 모두 정규분포를 따르면 등분산성 검정을 수행합니다. 등분산성을 만족하는 경우, 독립표본 t-검정을 수행하고, 등분산성을 만족하지 않는 경우에는 등분산이 아닌 독립표본 t-검정을 수행합니다. 만약 양쪽 그룹 모두 정규분포를 따르지 않는 경우, 비모수적인 맨-위트니 U 검정을 수행합니다.
- 각 A/B 테스트 결과는 입력된 그룹 변수, 테스트 유형, 귀무가설 기각 여부, p-값, 설명, A 그룹 평균 / 중앙값, B 그룹 평균/중앙값 등의 정보를 담고 있는 데이터 프레임으로 출력됩니다. 출력된 A/B 테스트 결과들은 모두 연결(concatenate)되어 최종 결과로 출력됩니다.

# 7. Time Related Features

시간 관련 특성

How many features can you create from only date column? I’m sharing an example of time related features. You can expand the features with your imagination or your needs

날짜 열만 있어도 얼마나 많은 특성을 만들 수 있을까요? 저는 시간 관련 특성의 예시를 공유하겠습니다. 여러분의 상상력이나 필요에 따라 이러한 특성을 더 확장해 볼 수 있습니다.

```python
# Time Related Features
def create_date_features(df):
    df['month'] = df.date.dt.month.astype("int8")
    df['day_of_month'] = df.date.dt.day.astype("int8")
    df['day_of_year'] = df.date.dt.dayofyear.astype("int16")
    df['week_of_month'] = (df.date.apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
    df['week_of_year'] = (df.date.dt.weekofyear).astype("int8")
    df['day_of_week'] = (df.date.dt.dayofweek + 1).astype("int8")
    df['year'] = df.date.dt.year.astype("int32")
    df["is_wknd"] = (df.date.dt.weekday // 4).astype("int8")
    df["quarter"] = df.date.dt.quarter.astype("int8")
    df['is_month_start'] = df.date.dt.is_month_start.astype("int8")
    df['is_month_end'] = df.date.dt.is_month_end.astype("int8")
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype("int8")
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype("int8")
    df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
    df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
    return df
d = create_date_features(d)

# Workday column
d["workday"] = np.where((d.holiday_national_binary == 1) | (d.holiday_local_binary==1) | (d.holiday_regional_binary==1) | (d['day_of_week'].isin([6,7])), 0, 1)
d["workday"] = pd.Series(np.where(d.IsWorkDay.notnull(), 1, d["workday"])).astype("int8")
d.drop("IsWorkDay", axis = 1, inplace = True)

# Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. 
# Supermarket sales could be affected by this.
d["wageday"] = pd.Series(np.where((d['is_month_end'] == 1) | (d["day_of_month"] == 15), 1, 0)).astype("int8")

d.head(15)
```

- 위 코드는 날짜 관련 데이터에서 다양한 feature들을 생성하는 함수입니다.
- 생성된 feature들은 다음과 같습니다.

![A-Comprehensive-Guide17](https://user-images.githubusercontent.com/130429032/235813468-7639f3f8-a573-4e55-a862-49d7ec1684b6.png)

# 8. Did Earthquake affect the store sales?

지진이 가게 매출에 영향을 미쳤습니까?

A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.

2016년 4월 16일 에콰도르에서 7.8 규모 지진이 발생했습니다. 사람들은 구호 물품으로 물 등을 기부하며 모금활동을 벌였고, 이로 인해 지진 이후 몇 주간 슈퍼마켓 판매에 큰 영향을 미쳤습니다.

Comparing average sales by year, month and product family will be one of the best ways to be able to understand how earthquake had affected the store sales.

해당 지진이 상점 판매에 어떤 영향을 미쳤는지 이해하기 위해서는 연도, 월, 제품군별 평균 판매 비교가 가장 좋은 방법 중 하나입니다.

We can use the data of March, April, May and June and there may be increasing or decreasing sales for some product families.

3월, 4월, 5월, 6월 데이터를 사용하여, 일부 제품군의 판매가 증가 또는 감소할 수 있습니다.

Lastly, we extracted a column for earthquake from Holidays and Events data.

“events_Terremoto_Manabi” column will help to fit a better model.

마지막으로, 휴일 및 이벤트 데이터에서 지진에 대한 열을 추출했습니다. “events_Terremoto_Manabi”열은 더 나은 모델링에 도움이 됩니다.

```python
d[(d.month.isin([4, 5]))].groupby(["year"]).sales.mean()
```

![A-Comprehensive-Guide18](https://user-images.githubusercontent.com/130429032/235813470-2673a37a-35fd-49f6-b8b8-521a84e8c3ef.png)

- 이 코드는 4월과 5월에 해당하는 월별 평균 판매량을 연도별로 그룹화하여 출력합니다. ‘d’ 데이터 프레임에서 월(month)이 4 또는 5인 데이터를 선택한 뒤, ‘year’열을 기준으로 그룹화하고, 이를 이용하여 ‘sales’열의 평균 값을 구합니다.

## March

```python
pd.pivot_table(d[(d.month.isin([3]))], index = "year", columns = "family", values = "sales", aggfunc = "mean")
```

![A-Comprehensive-Guide19](https://user-images.githubusercontent.com/130429032/235813473-2e94081e-87cd-46dc-bdaf-647777296d0f.png)

## April - May

```python
pd.pivot_table(d[(d.month.isin([4,5]))], index="year", columns="family", values="sales", aggfunc="mean")
```

![A-Comprehensive-Guide20](https://user-images.githubusercontent.com/130429032/235813474-79aaac07-160f-4458-bc28-2a95cf1d6c4e.png)

## June

```python
pd.pivot_table(d[(d.month.isin([6]))], index="year", columns="family", values="sales", aggfunc="mean")
```

![A-Comprehensive-Guide21](https://user-images.githubusercontent.com/130429032/235813476-9567da5b-fec5-4226-ac27-177d2e83bc5d.png)

# 9. ACF & PACF for each family

제품군별 ACF & PACF

The lag features means, shifting a time series forward one step or more than one. So, a lag feature can use in the model to improve it. However, how many lag features should be inside the model? For understanding that, we can use ACF and PACF. The PACF is very useful to decide which features should select.

lag 특성이란, 시계열 데이터를 한 단계 또는 여러 단계 앞으로 이동시키는 것을 의미합니다. 따라서 모델에 lag 특성을 추가하여 모델을 개선할 수 있습니다. 그러나, 몇 개의 lag 특성을 모델 안에 포함해야 하는지 어떻게 결정할까요? 이를 이해하기 위해, ACF와 PACF를 사용할 수 있습니다. PACF는 선택해야 할 특성을 결정하는 데 매우 유용합니다.

In our problem, we have multiple time series and each time series have different pattern of course. You know that those time series consists of store-product family combinations and we have 54 stores and 33 product families. We can’t examine all of them one by one. For this reason, I will look at average sales for each product but it will be store independent.

우리 문제에서는, 다양한 패턴을 가진 여러 개의 시계열 데이터가 있습니다. 각 시계열 데이터는 가게-제품군 조합으로 구성되며, 54개의 가게와 33개의 제품군이 있습니다. 이를 일일이 검사할 수 없습니다. 따라서 각 제품 패밀리의 평균 매출을 살펴보겠습니다. 이는 가게에 독립적입니다.

In addition, the test data contains 15 days for each family. We should be careful when selecting lag features. We can’t create new lag features from 1 lag to 15 lag. It must be starting 16.

또한, 테스트 데이터는 각 패밀리에 대해 15일씩을 포함합니다. lag 특성을 선택할 때 주의해야 합니다. 1 lag부터 15 lag까지 새로운 lag 특성을 생성할 수 없습니다. 16부터 시작해야 합니다.

```python
#a = d[d["store_nbr"] == 1].set_index("date")
a = d[(d.sales.notnull())].groupby(["date", "family"]).sales.mean().reset_index().set_index("date")
for num i in enumerate(a.family.unique()):
		try:
				fig, ax = plt.subplots(1, 2, figsize = (15, 5))
				temp = a[(a.family == i)]
				sm.graphics.tsa.plot_acf(temp.sales, lags = 365, ax = ax[0], title = "ATUOCORRELATION\n" + i)
				sm.graphics.tsa.plot_pacf(temp.sales, lags = 365, ax = ax[1], title = "PARTIAL AUTOCORRELATION\n" + i)
		except:
				pass
```

![A-Comprehensive-Guide22](https://user-images.githubusercontent.com/130429032/235813477-07341345-b4e1-4e8d-ab3a-6874c94fe10f.png)

- 이 코드는 시계열 데이터에서 각 제품군별로 자기상관 및 부분자기상관함수를 시각화하는 기능을 합니다.
- 데이터셋에서 먼저 “sales” 열 값이 null이 아닌 데이터들을 선택하고, 날짜(date)와 제품군(family)을 기준으로 평균 매출을 계산합니다. 그리고 이를 다시 날짜(date)를 인덱스로 설정하여 a 변수에 저장합니다.
- 그 다음 for 문을 통해 각 제품군(family)을 unique()함수를 사용하여 추출합니다. 그리고 try - except 구문을 사용하여 해당 제품군(family) 데이터가 없는 경우를 처리합니다.
- try 안에서는 해당 제품군(family)에 대한 시계열 자기상관함수(ACF)와 부분자기상관함수(PACF)를 계산하고 시각화합니다. lags = 365로 설정하면 1년 단위로 시차(lag)를 나타낼 수 있습니다. 두 그래프 모두 x축은 시차(lag), y축은 상관관계 값을 나타냅니다. ACF는 전체 lag 범위에 대한 자기상관 값을, PACF는 해당 lag와 다른 lag 사이에서 관련성을 보이는 상관성 값을 나타냅니다.

I decided to chose these lags 16, 20, 30, 45, 365, 730 from PACF. I don’t know that they will help me to improve the model but especially, 365th and 730th lags may be helpful. If you compare 2016 and 2017 years for sales, you can see that they are highly correlated.

PACF에서 16, 20, 30, 45, 365, 730 이라는 시차(lag)를 선택하기로 결정했습니다. 이들이 모델을 개선하는 데 도움이 될지는 모르지만, 특히 365번째와 730번째 시차는 유용할 수 있습니다. 매출에 대해 2016년과 2017년을 비교해보면, 높은 상관관계를 보입니다.

```python
a = d[d.year.isin([2016, 2017]).groupby(["year", "day_of_year"])].sales.mean().reset_index()
px.line(a, x = "day_of_year", y = "sales", color = "year", title = "Average sales for 2016 and 2017")
```

![A-Comprehensive-Guide23](https://user-images.githubusercontent.com/130429032/235813478-83a0befa-8b20-4bb5-86b1-991d164dfc6e.png)

# 10. Simple Moving Average

```python
a = train.sort_values(["store_nbr", "family", "date"])
for i in [20, 30, 45, 60, 90, 120, 365, 730]:
		a["SMA" + str(i) + "_sales_lag16"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(16).values
		a["SMA" + str(i) + "_sales_lag30"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(30).values
		a["SMA" + str(i) + "_sales_lag60"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(60).values
print("Correlation")
a[["sales"] + a.columns[a.columns.str.startswith("SMA"].tolist()].corr()
```

![A-Comprehensive-Guide24](https://user-images.githubusercontent.com/130429032/235813411-74393ff4-6519-46d6-93e8-0b6dfd86d320.png)

- 이 코드는 train 데이터프레임에 대해 각 매장(store_nbr)과 제품군(family)별로 이동평균(SMA)을 구하고, 이를 이용하여 매출(sales)과의 상관관계를 구하는 것을 목적으로 합니다.
- 먼저 train 데이터프레임을 ‘store_nbr’, ‘family’, ‘date’ 컬럼을 기준으로 정렬합니다. 그리고 이동평균(SMA)을 구하기 위해 for 루플 사용합니다. 루프 내에서는 SMA 값을 구하고 16, 30, 60일 전의 SMA값을 해당 열에 저장합니다.
- 예를 들어, “SMA20_sales_lag16”은 16일 전부터 20일간의 이동평균 매출(sales)을 나타내는 컬럼이 됩니다. 이와 같은 방식으로 SMA 값을 계산하고 해당 열에 저장합니다.
- 마지막으로, “sales”과 “SMA”로 시작하는 컬럼들 사이의 상관관계를 구하기 위해 corr()함수를 사용합니다. 이를 통해 매출(sales)과 이동평균(SMA)간의 상관관계를 계산하고 출력합니다.

```python
b = a[(a.store_nbr == 1)].set_index("date")
for i in b.family.unique():
    fig, ax = plt.subplots(2, 4, figsize = (20, 10))
    b[b.family == i][["sales", "SMA20_sales_lag16"]].plot(legend = True, ax = ax[0, 0], linewidth = 4)
    b[b.family == i][["sales", "SMA30_sales_lag16"]].plot(legend = True, ax = ax[0, 1], linewidth = 4)
    b[b.family == i][["sales", "SMA45_sales_lag16"]].plot(legend = True, ax = ax[0, 2], linewidth = 4)
    b[b.family == i][["sales", "SMA60_sales_lag16"]].plot(legend = True, ax = ax[0, 3], linewidth = 4)
    b[b.family == i][["sales", "SMA90_sales_lag16"]].plot(legend = True, ax = ax[1, 0], linewidth = 4)
    b[b.family == i][["sales", "SMA120_sales_lag16"]].plot(legend = True, ax = ax[1, 1], linewidth = 4)
    b[b.family == i][["sales", "SMA365_sales_lag16"]].plot(legend = True, ax = ax[1, 2], linewidth = 4)
    b[b.family == i][["sales", "SMA20_sales_lag16"]].plot(legend = True, ax = ax[1, 3], linewidth = 4)
    plt.suptitle("STORE 1 - " + i, fontsize = 15)
    plt.tight_layout(pad = 1.5)
    for j in range(0, 4):
        ax[0, j].legend(fontsize = "x-large")
        ax[1, j].legend(fontsize = "x-large")
    plt.show()
```

![A-Comprehensive-Guide25](https://user-images.githubusercontent.com/130429032/235813414-24d13178-9c10-4bb3-8783-1b18ed2b2ffd.png)

- 이 코드는 store_nbr이 1인 상점에서 판매되는 제품군에 대해 SMA와 sales 간의 관계를 시각화합니다. 각 제품군별로 SMA20, SMA30, SMA45, SMA60, SMA90, SMA120, SMA365, SMA730의 8개의 SMA(lag16, 즉 16일 이전부터 SMA 값을 사용한다)를 시계열 그래프로 그려줍니다. 이를 통해, SMA의 변화가 sales에 미치는 영향을 시각적으로 파악할 수 있습니다. 또한, 각 그래프에는 판매량과 SMA의 값을 나란히 표시하여 비교할 수 있도록 합니다.

# 11. Exponential Moving Average

```python
def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store_nbr", "family"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha = alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [16, 30, 60, 90]

a = ewm_features(a, alphas, lags)
```

- 이 코드는 지수 가중 이동 평균(Exponential Weighted Moving Average, EWMA)을 사용하여 feature 엔지니어링을 수행합니다. 이를 통해 주어진 시계열 데이터에 대한 지수 가중 이동 평균을 계산하고, 이를 이용하여 더 많은 feature를 생성합니다.
- ‘ewm_features’ 함수는 데이터프레임과 알파 값 리스트, lag 값 리스트를 인자로 받습니다. 인자로 받은 알파 값 리스트와 lag 값 리스트에 따라서 각각의 조합에 대해 지수 가중 이동 평균을 계산하고, 이를 기존 데이터프레임에 추가합니다.
- 이 함수를 실행하기 전에, 이 코드에선 이전에 생성한 데이터프레임을 ‘a’ 라는 이름으로 사용하고 있으며 ‘a’ 데이터프레임에는 이미 이전 단계에서 ‘SMA’ 피처가 추가되어 있습니다.

```python
a[(a.store_nbr == 1) & (a.family == "GROCERY I")].set_index("date")[["sales", "sales_ewm_alpha_095_lag_16"]].plot(title = "STORE 1 - GROCERY I")
```

![A-Comprehensive-Guide26](https://user-images.githubusercontent.com/130429032/235813416-7ac08c6e-8fbb-4232-b38b-6f78951e0b0d.png)