---
layout: single
title:  "Store Sales. Time Series Forecast & Visualization 2"
categories: Transcription
tag: [Python, Kaggle, Store, Sales, Transcription]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

매장 매출. 시계열 예측 및 시각화

# 3. Visualize data

데이터 시각화

one of the biggest parts of the notebook. Here we can look through some variables and see some dependencies. Firstly, let’s check the dependency of the oil from the data:

노트북에서 가장 큰 부분 중 하나입니다. 여기서는 몇 가지 변수를 살펴보고 종속성을 확인할 수 있습니다. 먼저, 날짜로부터 기름의 종속성을 확인해보겠습니다.

```python
fig,axes = plt.subplots(nrows = 1, ncols = 1, figsize = (25, 15))
df_oil.plot.line(x = "date", y = "dcoilwtico", color = 'b', title = "dcoilwtico", ax = axes, rot = 0)
plt.show()
```

![time-series-fv-2-](https://user-images.githubusercontent.com/130429032/236437673-9eb3f3ed-b072-44ae-a04e-102ed2051fa3.png)

- ‘**rot’** 인자는 x축 레이블을 회전시키는 정도를 의미합니다.

As we have so much rows in out dataset, it wiil be easier to group data, as example, by week or month. The aggreagation will be made by mean.

데이터셋에 행이 너무 많으므로 주 또는 월별로 데이터를 그룹화하는 것이 더 쉬워집니다. 집계는 평균으로 이루어집니다.

```python
def grouped(df, key, freq, col):
		""" GROUP DATA WITH CERTAIN FREQUENCE """
		df_grouped = df.groupby([pd.Grouper(key = key, freq = freq)]).agg(mean = (col, 'mean'))
		df_grouped = df.grouped.reset_index()
		return df_grouped
```

- 이 코드는 ‘pandas’ 라이브러리를 사용하여 데이터프레임을 그룹화하고, 그룹화된 데이터를 지정된 주기(’freq’)로 집계하는 함수입니다.
- ‘df’는 그룹화할 데이터가 포함된 데이터프레임입니다. ‘key’는 그룹화할 기준이 되는 열의 이름이며, ‘col’은 집계할 대상이 되는 열의 이름입니다. ‘freq’는 그룹화할 주기를 나타내는 문자열입니다.
- ‘groupby’ 메소드를 사용하여 데이터를 그룹화하고, ‘agg’ 메소드를 사용하여 그룹화된 데이터를 집계합니다. ‘agg’ 메소드에서는 ‘mean’ 함수를 사용하여 ‘col’열의 평균값을 계산하고, ‘reset_index’ 메소드를 사용하여 그룹화된 데이터를 새로운 데이터프레임으로 반환합니다.
- ‘pd.Grouper()’는 ‘pandas’ 라이브러리에서 제공하는 시간 데이터를 그룹화하기 위한 유틸리티 함수입니다. 시계열 데이터를 다룰 때, 데이터를 일정한 주기로 묶어서 집계하고자 할 때 사용됩니다. 인자로 전달된 ‘key’와 freq’를 기준으로 데이터를 그룹화합니다. ‘key’는 그룹화할 기준이 되는 열의 이름이며, ‘freq’는 그룹화할 주기를 나타내는 문자열입니다. ‘groupby’ 메소드와 함께 사용하며, 그룹화한 데이터에 적용시켜 시간 간격(’freq’)에 따라 데이터를 그룹화할 수 있습니다.

Here we can check grouped data:

```python
# check grouped data
df_grouped_trans_w = grouped(df_trans, 'date', 'W', 'transactions')
df_grouped_trans_w
```

![time-series-fv-2-1](https://user-images.githubusercontent.com/130429032/236437709-fa085f58-e541-4ffb-8271-9ae693967ffe.png)

And, for better forecasting we’ll add ‘time’ column to our dataframe.

```python
def add_time(df, key, freq, col):
		""" ADD COLUMN 'TIME' TO DF """
		df.grouped = grouped(df, key, freq, col)
		df_grouped['time'] = np.arange(len(df_grouped.index))
		column_time = df_grouped.pop('time')
		df_grouped.insert(1, 'time', column_time)
		return df_grouped
```

- 이 코드는 데이터프레임에 ‘time’ 열을 추가하는 함수입니다.
- ‘df’는 ‘key’열을 기준으로 그룹화된 데이터프레임입니다. ‘key’는 그룹화할 기준이 되는 열의 이름이며, ‘col’은 집계할 대상이 되는 열의 이름입니다. ‘freq’는 그룹화할 주기를 나타내는 문자열입니다.
- ‘grouped()’ 함수를 사용하여 데이터를 그룹화하고, ‘np.arange()’함수를 사용하여 ‘time’열에 시간 정보를 추가합니다. ‘pop()’ 메소드를 사용하여 ‘time’열을 추출한 후, ‘insert()’ 메소드를 사용하여 ‘time’ 열을 새로운 위치에 추가합니다.

So, now we can check the results of grouping on the example of df_train (grouped by weeks on sales, after that, mean was counted).

이제 df_train 예제에서 그룹화 결과를 확인할 수 있습니다.(매출 주 단위로 그룹화 후 평균을 계산)

```python
df_grouped_train_w = add_time(df_train, 'date', 'W', 'sales')
df_grouped_train_m = add_time(df_train, 'date', 'M', 'sales')

df_grouped_train_w.head() # check results
```

![time-series-fv-2-2](https://user-images.githubusercontent.com/130429032/236437718-4ed789e2-dee8-4ef9-bfe2-5b155dfff403.png)

# 3.1. Linear Regression

선형 회귀 분석

After that, we can build some more plots. Linear regression is widely used in practice and adapts naturally to even complex forecasting tasks. The linear regression algorithm learns how to make a weighted sum from its input features.

그런 다음, 몇 가지 플롯을 만들 수 있습니다. 선형 회귀는 실무에서 널리 사용되며 복잡학 예측 작업에도 자연스럽게 적응합니다. 선형 회귀 알고리즘은 입력된 features에서 가중 합계를 만드는 방법을 학습합니다.

```python
fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (30, 20))

# TRANSACTIONS (WEEKLY)
axes[0].plot('date', 'mean', data = df_grouped_trans_w, color = 'grey', marker = 'o')
axes[0].set_title("Transactions (grouped by week)", fontsize = 20)

# SALES (WEEKLY)
axes[1].plot('time', 'mean', data = df_grouped_train_w, color = '0.75')
axes[1].set_title("Sales (grouped by week)", fontsize = 20)

# linear regression
axes[1] = sns.regplot(x = 'time', y = 'mean', data = df_grouped_train_w, scatter_kws = dict(color = '0.75'), ax = axes[1])

# SALES (MONTHLY)
axes[2].plot('time', 'mean', data = df_grouped_train_m, color = '0.75')
axes[2].set_title("Sales (grouped by month)", fontsize = 20)

# linear regression
axes[2] = sns.regplot(x = 'time', y = 'mean', data = df_grouped_train_m, scatter_kws = dict(color = '0.75'), line_kws = {"color": "red"}, ax = axes[2])
plt.show()
```

![time-series-fv-2-3](https://user-images.githubusercontent.com/130429032/236437724-6985abaf-c79e-4bba-9b71-18658ddbffbd.png)

- sns.regplot() 함수는 seaborn 패키지에서 제공하는 선형 회귀 분석을 수행하고 결과를 시각화하는 함수입니다.
- ‘x’와 ‘y’ 사이의 선형 관계를 시각화하기 위해 산점도와 함께 회귀선을 그립니다. 또한 그래프에서 플롯된 데이터 포인트 주변에 회귀선과의 거리에 따라 생삭을 달리하여 회귀선의 적합도를 시각적으로 확인할 수 있도록 합니다.
- ‘scatter_kws’는 산점도에 적용할 스타일을 지정하는 인자입니다. 딕셔너리 형태로 전달되며, 여기선 ‘color’를 0.75로 지정하여 회색조의 점들을 그리도록 했습니다.
- ‘line_kws’는 회귀선에 적용할 스타일을 지정하는 인자입니다. 이번엔 딕셔너리 형태로 전달되며 ‘color’를 ‘red’로 지정하여 빨간색으로 회귀선을 그리도록 했습니다.

# 3.2 Lag feature

To make a lag feature we shift the observations of the target series so that they appear to have occured later in time. Here we’ve created a 1-step lag feature, though shifting by multiple steps is possible too. So, firstly, we should add lag to our data.

지연 feature를 만들려면 대상 계열의 관측값을 이동하여 나중에 발생한 것처럼 보이도록 해야합니다. 여기서는 1단계 지연 feature를 만들었지만 여러 단계로 이동하는 것도 가능합니다. 따라서 먼저 데이터에 지연을 추가해야합니다.

```python
def add_lag(df, key, freq, col, lag):
		""" ADD LAG """
		df.grouped = grouped(df, key, freq, col)
		name = 'Lag_' + str(lag)
		df_grouped['Lag'] = df_grouped['mean'].shift(lag)
		return df_grouped
```

- ‘add_lag()’ 함수는 입력된 데이터프레임에 대해 특정 시간 간격(’freq’)에 대해 그룹화하고, 해당 그룹의 특정 열(’col’)에 대한 평균 값을 계산합니다. 그리고 ‘lag’ 값만큼 시간을 이동시켜 새로운 열(’Lag’)를 생성합니다.
- 즉, 이 함수는 시계열 데이터에서 자주 사용하는 레깅(lagging)작업을 수행하는 함수입니다. 래깅은 이전 시간의 데이터를 현재 시간에 사용하는 것을 의미합니다. 이를 통해 시간 간격별로 데이터의 패턴을 파악하거나, 예측 모델링을 위한 변수를 생성하는 등 다양한 분석에 활용될 수 있습니다.

Here we can check grouped data with lag:

```python
df_grouped_train_w_lag1 = add_lag(df_train, 'date', 'W', 'sales', 1)
df_grouped_train_m_lag1 = add_lag(df_train, 'date', 'M', 'sales', 1)

df_grouped_train_w_lag1.head() # check data
```

![time-series-fv-2-4](https://user-images.githubusercontent.com/130429032/236437741-c011fd1d-768f-4e92-812c-484ad614da46.png)

So lag features let us fit curves to lag plots where each observation in a series is plotted against the previous observation. Let’s build same plots, but with ‘lag’feature:

```python
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (30, 20))
axes[0].plot('Lag', 'mean', data = df_grouped_train_w_lag1, color = '0.25', linestyle = (0, (1, 10)))
axes[0].set_title("Sales (grouped by week)", fontsize = 20)
axes[0] = sns.regplot(x = 'Lag', y = 'mean', data = df_grouped_train_w_lag1, scatter_kws = dict(color = '0.25'), ax = axes[0])

axes[1].plot('Lag', 'mean', data = df_grouped_train_m_lag1, color = '0.25', linestyle = (0, (1, 10)))
axes[1].set_title("Sales (grouped by month)", fontsize = 20)
axes[1] = sns.regplot(x = 'Lag', y = 'mean', data = df_grouped_train_m_lag1, scatter_kws = dict(color = '0.25'), line_kws = {"color": "red"}, ax = axes[1])
plt.show()
```

![time-series-fv-2-5](https://user-images.githubusercontent.com/130429032/236437744-4e239149-62fe-4cd5-b2bb-d7e88a0ca5b3.png)

- 이 그래프는 만매량과 그 전 주 또는 월 판매량 간의 관계를 보여주고 있어요.
- x축은 전 주 또는 월 판매량을, y축은 이번 주 또는 월 판매량을 나타내고 있습니다. 빨간색 선은 판매량과 이전 판매량 간의 관계를 보여주는 선형 회귀선입니다.
- 점들은 판매량 데이터를 나타내며, 회색 선은 점들 사이의 추세를 나타냅니다. 점들이 회색 선 주변에 모여 있으면 판매량이 큰 차이가 없는 것이고, 점들이 멀리 흩어져 있으면 판매량 차이가 큰 것입니다.

# 3.3 Some more statistics & visualizations

더 많은 통계 및 시각화

In this block we are going to explore data. Firstly, let’s count for each category in each dataset, value_counts():

이 블록에서는 데이터를 탐색해 보겠습니다. 먼저, 각 데이터 집합의 각 카테고리에 대해 value_counts() 함수를 사용해 카운트 해보겠습니다.

```python
def plot_stats(df, column, ax, color, angle):
		""" PLOT STATS OF DIFFERENT COLUMNS """
		count_classes = df[column].value_counts()
		ax = sns.barplot(x = count_classes.index, y = count_classes, ax = ax, palette = color)
		ax.set_title(column.upper(), fontsize = 18)
		for tick in ax.get_xticklabels():
				tick.set_rotation(angle)
```

- 이 함수는 데이터프레임의 특정 컬럼(column)의 값들을 분석하여 그래프로 표시하는 함수입니다.
- 먼저, value_counts() 함수를 사용하여 각 값이 몇 번 나왔는지 계산하고, barplot() 함수를 사용하여 x축에는 값의 종류, y축에는 각 값이 나온 횟수(count)를 나타내는 막대 그래프(bar graph)를 그립니다.
- 마지막으로 x축 라벨을 angle 매개변수에 지정된 각도로 회전시켜 표시합니다. 이는 x축 라벨이 많이 겹치는 경우 가독성을 높이기 위한 방법입니다.

```python
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
fig.autofmt_xdate()
fig.suptitle("Stats of df_holidays".upper())
plot_stats(df_holidays, "type", axes[0], "pastel", 45)
plot_stats(df_holidays, "locale", axes[1], "rocket", 45)
plt.show()
```

![time-series-fv-2-6](https://user-images.githubusercontent.com/130429032/236437758-4f536087-595f-4a7e-82ae-fbd8315624db.png)

- ‘fig.autofmt_xdate()’ 함수는 x축 레이블의 날짜 형식을 자동으로 맞춰서 보기 좋게 설정해주는 함수입니다.

```python
fig, axes = plt.subplots(nrows = 4, ncols = 1, figsize = (20, 4))
plot_stats(df_stores, "city", axes[0], "mako_r", 45)
plot_stats(df_stores, "state", axes[1], "rocket_r", 45)
plot_stats(df_stores, "type", axes[2], "magma", 0)
plot_stats(df_stores, "cluster", axes[3], "viridis", 0)
plt.show()
```

![time-series-fv-2-7](https://user-images.githubusercontent.com/130429032/236437760-e41fa1df-93b4-4f82-897d-772028c525fe.png)

Let’s plot pie chart for ‘family’ of df_train:

```python
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 10))
count_classes = df_train['family'].value_counts()
plt.title("Stats of df_train".upper())
colors = ['#ff9999', '#66b3ff', '#99ff99',
					'#ffcc99', '#ffccf9', '#ff99f8',
					'#ff99af', '#ffe299', '#a8ff99',
					'#cc99ff', '#9e99ff', '#99c9ff',
					'#99f5ff', '#99ffe4', '#99ffaf']

plt.pie(count_classes, labels = count_classes.index, autopct = '%1.1f%%', shadow = True, startangle = 90, colors = colors)

plt.show()
```

![time-series-fv-2-8](https://user-images.githubusercontent.com/130429032/236437771-afcd0f03-49fe-448f-87ca-138df10af7ef.png)

# 3.4 BoxPlot

In addition, we can build some boxplots: for df_oul & df_trans.

```python
def plot_boxplot(palette, x, y, hue, ax, title):
		sns.set_theme(style = "ticks", palette = palette)
		ax = sns.boxplot(x = x, y = y, hue = hue, ax = ax)
		ax.set_title(title, fontsize = 18)
```

- ‘sns.set_theme()’ 함수는 seaborn 패키지에서 제공하는 스타일과 색상 등의 설정을 변경할 수 있습니다.

```python
fig, axes = plt.subplots(nrows = 4, ncols = 1, figsize = (30, 60))
plot_boxplot("pastel", df_oil['date'].dt.year, df_oil['dcoilwtico'], df_oil['date'].dt.month, axes[0], "df_oil")
plot_boxplot("pastel", df_oil['date'].dt.year, df_oil['dcoilwtico'], df_oil['date'].dt.year, axes[1], "df_oil")
plot_boxplot("pastel", df_trans['date'].dt.year, df_trans['transactions'], df_trans['date'].dt.month, axes[2], "df_trans")
plot_boxplot("pastel", df_trans['date'].dt.year, df_trans['transactions'], df_trans['date'].dt.year, axes[3], "df_trans")
plt.show()
```

![time-series-fv-2-9](https://user-images.githubusercontent.com/130429032/236437787-3883a39c-3088-4491-b5dc-d9f2f44f8354.png)

# 3.5 Trend. Moving Average

추세. 이동 평균

The trend component of a time series represents a persistent, long-term change in the mean of the series. The trend is the slowest-moving part of a series, the part representing the largest time scale of importance. In a time series of product sales, an increasing trend might be the effect of a market expansion as more people become aware of the product year by year.

시계열의 추세 성분은 시리즈의 평균값에 대한 지속적이고 장기적인 변화를 나타냅니다. 추세는 시리즈에서 가장 느리게 움직이는 부분으로서, 가장 중요한 시간 척도를 나타냅니다. 제품 판매의 시계열에서 증가하는 추세는 제품에 대해 매년 더 많은 사람들이 인식함에 따른 시장 확장의 영향으로 해석될 수 있습니다.

To see what kind of trend a time series might have, we can use a moving average plot. To compute a moving average of a time series, we compute the average of the values within a sliding window of some defined width. Each point on the graph represents the average of all the values in the series that fall within the window on either side. The idea is to smooth out any short-term fluctuations in the series so that only long-term changes remain.

어떤 종류의 추세가 있는지 확인하기 위해 이동 평균 그래프를 사용할 수 있습니다. 시계열의 이동 평균을 계산하려면, 정의된 너비의 슬라이딩 창 내의 값들의 평균을 계산합니다. 그래프의 각 지점은 창 양쪽에 속하는 시리즈의 모든 값의 평균을 나타냅니다. 아이디어는 시리즈의 단기적인 변동을 평준화하여 장기적인 변화만 남기는 것입니다.

Below we can see the moving average plots for Transactions and Sales, colored in green.

아래 그림에서는 거래 및 판에 대한 이동 평균 그래프를 볼 수 있습니다. 이 그래프는 초록색으로 표시되어 있습니다.

```python
def plot_moving_average(df, key, freq, col, window, minperiods, ax, title):
		df_grouped = grouped(df, key, freq, col)
		moving_average = df_grouped['mean'].rolling(window = window, center = True, min_periods = min_periods).mean()
		ax = df_grouped['mean'].plot(color = '0.75', linestyle = 'dashdot', ax = ax)
		ax = moving_average.plot(linewidth = 3, color = 'g', ax = ax)
		ax.set_title(title, fontsize = 18)
```

- 이 함수는 입력으로 데이터프레임, 그룹화할 키, 빈도, 열, 이동평균 계산에 사용될 윈도우 크기, 최소 기간, 그리고 축과 제목을 받습니다.
- 이 함수를 사용하면 시계열 데이터의 추세를 파악할 수 있는데, 이동평균 값을 계산하여 시계열 데이터에서 발생하는 잡음과 변동을 완화시키고, 추세나 패턴 등을 파악하는 데 도움이 됩니다.

```python
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (30, 20))
plot_moving_average(df_trans, 'date', 'W', 'transactions', 7, 4, axes[0], 'Transactions Moving Average')
plot_moving_average(df_train, 'date', 'W', 'sales', 7, 4, axes[1], 'Sales Moving Average')
plt.show()
```

![time-series-fv-2-10](https://user-images.githubusercontent.com/130429032/236437817-362bea72-03c4-490a-8887-01b2805e8e00.png)

# 3.6 Trend. Forecasting Trend

추세. 추세 예측

We’ll use a function from the statsmodels library called DeterministicProcess. Using this function will help us avoid some tricky failure cases that can arise with time series and linear regression. The order argument refers to polynomial order: 1 for linear, 2 for quadratic, 3 for cubic and so on.

우리는 statsmodels 라이브러리에서 DeterministicProcess라는 함수를 사용할 것입니다. 이 함수를 사용하면 시계열과 선형 회귀에서 발생할 수 있는 어려운 실패 케이스를 피할 수 있습니다. order 인자는 다항식 차수를 의미합니다. 1은 선형, 2는 이차, 3은 삼차 등입니다.

```python
def plot_deterministic_process(df, key, freq, col, ax1, title1, ax2, title2):
    df_grouped = grouped(df, key, freq, col)
    df_grouped['date'] = pd.to_datetime(df_grouped['date'], format = "%Y-%m-%d")
    dp = DeterministicProcess(index = df_grouped['date'], constant = True, order = 1, drop = True)
    dp.index.freq = freq # manually set the frequency of the index
    # 'in_sample' creates features for the dates given in the 'index' argument
    X1 = dp.in_sample()
    y1 = df_grouped["mean"] # the target
    y1.index = X1.index
    # The intercept is the same sa the 'const' feature from
    # DeterministicProcess. LinearRegression behaves badly with duplicated
    # features, so we need to be sure to exclude it here.
    model = LinearRegression(fit_intercept = False)
    model.fit(X1, y1)
    y1_pred = pd.Series(model.predict(X1), index = X1.index)
    ax1 = y1.plot(linestyle = 'dashed', label = "mean", color = "0.75", ax = ax1, use_index = True)
    ax1 = y1_pred.plot(linewidth = 3, label = "Trend", color = 'b', ax = ax1, use_index = True)
    ax1.set_title(title1, fontsize = 18)
    _ = ax1.legend()

    # forecast Trend for future 30 steps
    steps = 30
    X2 = dp.out_of_sample(steps = steps)
    y2_fore = pd.Series(model.predict(X2), index = X2.index)
    y2_fore.head()
    ax2 = y1.plot(linestyle = 'dashed', label = "mean", color = "0.75", ax = ax2, use_index = True)
    ax2 = y1_pred.plot(linewidth = 3, label = "Trend", color = 'b', ax = ax2, use_index = True)
    ax2 = y2_fore.plot(linewidth = 3, label = "Predicted Trend", color = 'r', ax = ax2, use_index = True)
    ax2.set_title(title2, fontsize = 18)
    _ = ax2.legend()
```

- ‘grouped’ 함수를 사용하여 ‘df’ 데이터를 ‘key’열로 그룹화하고, ‘col’열의 평균을 구합니다.
- 그룹화된 데이터를 ‘pd.to_datetime’ 함수를 사용하여 날짜 형식으로 변환합니다.
- ‘DeterministicProcess’ 객체를 생성하여 추세 모델링을 수행합니다. ‘constant = True’로 설정하면 절편을 포함한 모델링을 수행합니다. ‘order = 1’로 설정하면 1차 차분을 수행하여 추세를 모델링합니다. ‘drop = True’ 로 설정하면 상수항(절편)을 제거합니다.
- ‘in_sample()’ 함수를 사용하여 추세 모델링에 사용할 feature를 생성합니다. ‘out_of_sample’ 함수를 사용하여 미래 30일의 feature를 생성합니다.
- 선형 회귀 모델(’LinearRegression’)을 생성하고, ‘in_Sample’ 함수로 생성된 feature와 ‘y1’ 데이터(평균)를 사용하여 모델을 훈련합니다.
- ‘predict’ 함수를 사용하여 추세를 예측합니다. 예측된 추세를 ‘plot’ 함수를 사용하여 그래프에 그립니다.
- 미래 30일의 추세를 예측하고, 예측된 추세를 ‘plot’ 함수를 사용하여 그래프에 그립니다.

Here we can see Linear Trend & Linear Trend Forecast for Transactions (plots 1, 2) and Sales (plots 3, 4).

```python
fig, axes = plt.subplots(nrows = 4, ncols = 1, figsize = (30, 30))
plot_deterministic_process(df_trans, 'date', 'W', 'transactions',
                           axes[0], "Transactions Linear Trend",
                           axes[1], "Transactions Linear Trend Forecast")
plot_deterministic_process(df_train, 'date', 'W', 'sales',
                           axes[2], "Sales Linear Trend",
                           axes[3], "Sales Linear Trend Forecast")
plt.show()
```

# 3.7 Seasonality

계절성

Time series exhibits seasonality whenever there is a regular, periodic change in the mean of the series. Seasonal changes generally follow the clock and calendar — repetitions over a day, a week, or a year are common. Seasonality is often driven by the cycles of the natural world over days and years or by conventions of social behavior surrounding dates and times. Just like we used a moving average plot to discover the trend in a series, we can use a seasonal plot to discover seasonal patterns.

시계열은 평균의 주기적인 변화가 있는 경우 계절성을 나타냅니다. 계절적인 변화는 일, 주 또는 연도에 걸쳐 반복되는 경향이 있습니다. 계절성은 일상적인 사이클과 연간 자연적인 세계의 변화 또는 날짜와 시간을 둘러싸고 있는 사회적 행동의 관례에 의해 일반적으로 추구됩니다. 시리즈에서 추세를 발견하기 위해 이동 평균 플롯을 사용한 것처럼, 계절 패턴을 발견하기 위해 계절성 플롯을 사용할 수 있습니다.

```python
def seasonal_plot(X, y, period, freq, ax = None):
    if ax is None:
        _, ax = plt.subplots() # ax가 None일 경우 새로운 figure와 axes 객체 생성
    palette = sns.color_palette("husl", n_colors = X[period].nunique(),)
    ax = sns.lineplot(x = X[freq],
                      y = X[y],
                      ax = ax,
                      hue = X[period],
                      palette = palette,
                      legend = False)
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(name,
                    xy = (1, y_),
                    xytext = (6, 0),
                    color = line.get_color(),
                    xycoords = ax.get_yaxis_transform(),
                    textcoords = "offset points",
                    size = 14,
                    va = "center")
        
    return ax
```

- 이 함수는 시계열 데이터에서 계절성(Seasonality)을 시각화하는 함수입니다.
- 인자로는 X(시계열 데이터), y(타겟 변수), period(계절성 주기를 나타내는 변수), freq(시간 주기를 나타내는 변수), 그리고 ax(축)를 받습니다.
- ‘sns.color_palette()’ 함수는 색상 파레트를 만드는 함수, ‘n_colors’는 생성할 색상의 개수를 정수로 전달합니다. 기본값은 6입니다.
- ‘nunique()’ 함수는 Pandas 라이브러리에서 제공하는 함수 중 하나로서, Series나 DataFrame 객체의 유일한 값의 개수를 반환합니다. 즉, 중복되지 않는 고유한 값의 수를 세는 함수입니다.
- ‘zip()’ 함수는 파이썬의 내장 함수로 여러 개의 리스트나 튜플 등의 iterable한 객체들의 요소를 묶어서 하나씩 차례로 반환하는 iterator를 만들어줍니다.
- ‘annotate()’ 메소드는 matplotlib의 기능 중 하나로, 그래프의 특정한 위치에 주석을 추가할 수 있습니다.
    - ‘xy’ : 주석의 위치를 지정하는 튜플 (x, y)
    - ‘xytext’ : 주석 텍스트의 위치를 지정하는 튜플 (x, y)
    - ‘textcoords’ : ‘xytext’의 좌표계를 지정합니다. 기본값은 “data”입니다.
    - ‘arrowprops’ : 화살표 속성을 지정하는 딕셔너리입니다. 화살표를 사용하지 않을 경우 None으로 설정합니다.
    - ‘fontsize’ : 주석 텍스트의 폰트 크기를 지정합니다.
    - ‘xycoords’와 ‘textcoords’는 어노테이션의 좌표계와 텍스트의 좌표계를 설정하는 인자입니다.
    - ‘va’는 텍스트의 수직 정렬을 설정하는 인자입니다.
- ‘get_ydata()’는 matplotlib의 ‘Line2D’ 객체에서 y값 데이터를 얻기 위한 메서드입니다. ‘Line2D’ 객체는 x, y값을 포함하고 있는 그래프의 선을 나타내는 객체입니다.

```python
def plot_periodogram(ts, detrend = 'linear', ax = None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(ts, fs = fs, detrend = detrend, window = "boxcar", scaling = 'spectrum')
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color = "purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(["Annual (1)", "Semiannual (2)", "Quarterly (4)",
                        "Bimonthly (6)", "Monthly (12)", "Biweekly ( 26)",
                        "Weekly (52)", "Semiweekly (104)"], rotation = 30) # 레이블을 30도 회전
    ax.ticklabel_format(axis = "y", style = "sci", scilimits = (0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax
```

- ts는 시계열 데이터를 의미합니다.
- pandas의 Timedelta는 시간 간격을 나타내는 객체로, 시간, 분, 초, 밀리초 등을 포함할 수 있습니다. 주로 시계열 분석에서 날짜 간의 차이나 간격을 계산하기 위해 사용됩니다.
- ‘periodogram()’ 함수는 주어진 시계열 데이터의 주파수 성분을 분석하여 주파수 대역별 파워 스펙트럼을 반환하는 함수입니다. 이는 신호처리에서 매우 중요한 분석 기법 중 하나입니다.
    - x: 입력 데이터(1차원 배열 형태)
    - fs: 입력 데이터의 샘플링 주파수
    - window: 윈도우 함수(기본값은 “boxcar”로 사각형 창 함수)
    - detrend: detrending 방법 (기본값은 “constant”로 절편 제거)
    - scaling 파워 스펙트럼 정규화 방식 (기본값은 ‘density’로 파워 스펙트럼의 총 합이 1이 되도록 조정)
- 주파수 대역과 파워 스펙트럼 값을 반환하는데, 이를 이용하여 시계열 데이터의 주요 주파수 대역과 그에 해당하는 신호 강도를 추정할 수 있습니다. 이를 통해, 신호가 어떤 주기로 변동하는지, 어떤 주기성 성분이 포함되어 있는지 등을 파악할 수 있습니다. 시계열 데이터의 주기성을 분석하거나, 주기성 성분을 제거하여 노이즈를 감소시키는 등의 목적으로 주로 사용됩니다.
- ax.step() 함수는 단계적인 그래프를 그릴 때 사용됩니다. 이 함수는 데이터의 값이 변화하는 지점에서 선을 갑작스럽게 꺾어서 그립니다.
    - x: x축 데이터
    - y: y축 데이터
    - where: “pre”, “post”, “mid” 중 하나로, x의 위치에 따라 단계의 위치를 결정합니다.
    - data: boolean 값으로 x, y의 데이터 위치를 결정합니다.
    - kwargs: 기타 그래프 스타일과 설정에 대한 파라미터
    - 보통 이산적인 데이터를 시각화할 때 사용됩니다.
- ax.set_xscale(”log”)는 x축의 스케일을 로그 스케일로 변경합니다. 로그 스케일은 값의 분포가 매우 크거나 작을 때 유용합니다.
- ax.ticklabel_format() 함수는 축의 눈금을 지정된 형식으로 변경하는 함수입니다. style은 지정된 형식 중 어떤 형식으로 눈금을 표시할 지 지정합니다. “sci”는 과학적 표기법으로 표시하는 형식입니다. scilimits는 과학적 표기법으로 표시할 때 지수 형태로 표현되는 최소/최대 자리수를 지정합니다.

```python
def seasonality(df, key, freq, col):
    df_grouped = grouped(df, key, freq, col)
    df_grouped['date'] = pd.to_datetime(df_grouped['date'], format = "%Y-%m-%d")
    df_grouped.index = df_grouped['date']
    df_grouped = df_grouped.drop(columns = ['date'])
    df_grouped.index.freq = freq # manually set the frequency of the index

    X = df_grouped.copy()
    X.index = pd.to_datetime(X.index, format = "%Y-%m-%d")
    X.index.freq = freq
    # days within a week
    X["day"] = X.index.dayofweek # the x-axis (freq)
    X["week"] = pd.Int64Index(X.index.isocalendar().week) # the seasonal period (period), 52주차
    # days within a year
    X["dayofyear"] = X.index.dayofyear
    X["year"] = X.index.year
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = (20, 30))
    seasonal_plot(X, y = 'mean', period = "week", freq = "day", ax = ax0)
    seasonal_plot(X, y = 'mean', period = "year", freq = "dayofyear", ax = ax1)
    X_new = (X['mean'].copy()).dropna()
    plot_periodogram(X_new, ax = ax2)
```

- ‘df_grouped.index = df_grouped['date']’는 index를 ‘date’열로 변경하는 코드입니다.
- ‘df_grouped.index.freq = freq’ 는 시계열 데이터에서 인덱스의 빈도(frequency)를 설정하는 속성입니다. 시계열 데이터에서는 일반적으로 데이터 포인트들 사이의 시간 간격(frequency)이 일정합니다. 이를 빈도(frequency)라고 하며, 예를 들어 일별 데이터에서는 빈도가 “1 day”이 될 수 있습니다.
- ‘isocalendar()’ 메소드는 인덱스에 있는 날짜들에 대해 ISO 캘린더(국제 표준화기구에서 제안한 달력 체계)의 연도, 주차, 요일을 반환하는 메소드입니다.
- ‘X.index.dayofyear’는 값이 해당 날짜가 해당 연도에서 몇 번째 날인지를 나타냅니다.

```python
# df_trans, grouped by day
seasonality(df_trans, 'date', 'D', 'transactions')
```

![time-series-fv-2-11](https://user-images.githubusercontent.com/130429032/236437852-97a5d9b0-a0ab-455d-b490-eca1017a0c0b.png)

Here we can see the plots for df_train.

```python
# df_train, grouped by day
seasonality(df_train, 'date', 'D', 'sales')
```

![time-series-fv-2-12](https://user-images.githubusercontent.com/130429032/236437888-3d13ea26-bc1c-40f9-acfb-ee2360eee586.png)

After that, we can predict seasonality, using DeterministicProcess, as we used for Trend. We are going to forecast seasonality for Transactions and Sales.

그런 다음 추세에 사용한 것처럼 결정론적 프로세스를 사용하여 계절성을 예측할 수 있습니다. 거래 및 매출에 대한 계절성을 예측해보겠습니다.

```python
def predict_seasonality(df, key, freq, col, ax1, title1):
    fourier = CalendarFourier(freq = "A", order = 10) # 10 sin/cos pairs for "A"nnual seasonality
    df_grouped = grouped(df, key, freq, col)
    df_grouped['date'] = pd.to_datetime(df_grouped['date'], format = "%Y-%m-%d")
    df_grouped['date'].freq = freq # manually set the frequency of the index
    dp = DeterministicProcess(index = df_grouped['date'],
                              constant = True,
                              order = 1,
                              period = None,
                              seasonal = True,
                              additional_terms = [fourier],
                              drop = True)
    dp.index.freq = freq # manually set the frequency of the index

    # 'in_sample' creates features for the dates given in the 'index' argument
    X1 = dp.in_sample()
    y1 = df_grouped["mean"] # the target
    y1.index = X1.index

    # The interceot is the same as the 'const' feature from
    # DeterministicProcess. LinearRegression behaves badly with duplicated
    # features, so we need to be sure to exclude it here.

    model = LinearRegression(fit_intercept = False) # 상수항을 고려하지 않습니다.
    model.fit(X1, y1)
    y1_pred = pd.Series(model.predict(X1), index = X1.index)
    X1_fore = dp.out_of_sample(steps = 90) # steps : 추가 관측치의 수
    y1_fore = pd.Series(model.predict(X1_fore), index = X1_fore.index)
    
    ax1 = y1.plot(linestyle = 'dashed', style = '.', label = "init mean values", color = "0.4", ax = ax1, use_index = True)
    ax1 = y1_pred.plot(linewidth = 3, label = "Seasonal", color = 'b', ax = ax1, use_index = True)
    ax1 = y1_fore.plot(linewidth = 3, label = "Seasonal Forecast", color = 'r', ax = ax1, use_index = True)
    ax1.set_title(title1, fontsize = 18)
    _ = ax1.legend()
```

- ‘CalendarFourier’는 statsmodels 패키지의 클래스 중 하나로, 주어진 시계열 데이터에서 연간 주기성을 모델링하는 데 사용됩니다. ‘freq’와 ‘order’ 매개변수를 사용하여 주기성의 주기와 반복 횟수를 지정할 수 있습니다. ‘freq’ 매개변수는 주기성을 나타내는 시간 단위를 나타내며, 기본값은 “A”로 연간 주기성을 의미합니다. “Q”로 지정하면 분기별 주기성을 모델링할 수 있고, “M”으로 지정하면 월별 주기성을 모델링할 수 있습니다. 이외에도 “W”와 같이 주 단위, “D”와 같이 일 단위 주기성을 모델링할 수 있습니다. “order” 매개변수는 주기성을 나타내는 삼각 함수의 반복 횟수를 결정합니다. 즉, 이 매개변수를 높일수록 더 많은 주기성 변동성을 포착할 수 있습니다.
- ‘DeterministicProcess’ 는 결정론적(deterministic) 프로세스를 모델링하는데 사용합니다.
    - ‘index’ 매개변수는 시계열 데이터의 인덱스를 지정합니다. 이 인덱스는 모델링에 사용되는 시간 변수로 사용됩니다.
    - ‘constant’ 매개변수는 모델에 상수항을 포함할지 여부를 결정합니다. ‘True’로 설정하면 상수항을 포함하고, ‘False’로 설정하면 상수항을 포함하지 않습니다.
    - ‘order’ 매개변수는 추세를 모델링하는 데 사용되는 다항식의 차수를 결정합니다. ‘order = 1’로 설정하면 일차 다항식으로 추세를 모델링하고, ‘order = 2’ 로 설정하면 이차 다항식으로 추세를 모델링합니다.
    - ‘period’ 매개변수는 모델링할 주기성을 지정합니다. ‘period = None’으로 설정하면 주기성을 모델링하지 않습니다.
    - ‘seasonal’ 매개변수는 모델링할 계절성을 지정합니다. ‘seasonal = True’로 설정하면 계절성을 모델링하고, ‘seasonal = False’로 설정하면 계절성을 모델링하지 않습니다.
    - ‘additional_terms’ 매개변수는 모델링할 추가적인 변동성을 지정합니다. 여기에서는 ‘fourier’와 같은 주기성 변동성을 추가합니다.
    - ‘drop’ 매개변수는 결측값을 처리하는 방법을 지정합니다. ‘drop = True’로 설정하면 결측값을 모두 제거하고, ‘drop = False’로 설정하면 결측값을 보간하여 처리합니다.
- ‘dp.in_sample()’은 객체 ‘dp’를 사용하여 학습된 예측 변수를 생성합니다. 이 메소드는 입력 데이터의 인덱스를 기반으로 하는 학습 데이터에서 생성된 예측 변수를 반환합니다.

```python
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (30, 30))
predict_seasonality(df_trans, 'date', 'W', 'transactions', axes[0], "Transactions Seasonal Forecast")
predict_seasonality(df_train, 'date', 'W', 'sales', axes[1], "Sales Seasonal Forecast")
plt.show()
```

![time-series-fv-2-13](https://user-images.githubusercontent.com/130429032/236437953-693ba352-c24d-4788-92e3-c8bf09c25fbc.png)