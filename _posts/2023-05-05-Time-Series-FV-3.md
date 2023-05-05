---
layout: single
title:  "Store Sales. Time Series Forecast & Visualization 3"
categories: Practice
tag: [Python, Kaggle, Store, Sales, Practice]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

매장 매출. 시계열 예측 및 시각화

# 4. Time Series as Features

```python
store_sales = df_train.copy()
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017', ['sales', 'onpromotion']]
)
mag_sales = family_sales.loc(axis=1)[:, 'MAGAZINES']
```

- **‘mag_sales’** 변수에는 2017년도의 각 날짜에 대해 ‘MAGAZINES’ 카테고리 제품의 판매 평균과 판매 중인 제품 비율(onpromotion)이 저장됩니다.
- **‘.unstack(’family’)’** 는 **‘family’** 열을 기준으로 데이터를 재배열하는 메소드입니다. 이 메소드를 사용하면 **‘family’** 열의 값이 새로운 컬럼 이름으로 사용됩니다.
- **‘loc(axis = 1)’** : 열에 대한 인덱싱을 수행합니다. **‘axis = 1’** 은 열 방향으로 인덱싱을 수행함을 의미합니다. 컬럼 레벨에서의 인덱싱을 수행할 수 있습니다.
- **‘[:, ‘MAGAZINES’]’** : 첫 번째 인덱서(’:’)는 모든 행(row)을 선택하고, 두 번째 인덱서(’MAGAZINES’)는 **‘family_sales’** 데이터 프레임에서 **‘MAGAZINES’** 카테고리에 해당하는 열만 선택하라는 의미입니다.

Here we can **check data store_sales**:

```python
store_sales.head()
```

![230505-FV](https://user-images.githubusercontent.com/130429032/236466114-c49eb6b9-77ee-4d8e-9120-5ddb78f882ec.png)

Here we can **check data mag_sales**:

```python
mag_sales.head()
```

![230505-FV1](https://user-images.githubusercontent.com/130429032/236466117-bc6aa5ca-df54-4920-872a-5842ba156e86.png)

Here we can **plot data**:

```python
y = mag_sales.loc[:, 'sales'].squeeze()

fourier = CalendarFourier(freq = 'M', order = 4)
dp = DeterministicProcess(
    constant = True,
    index = y.index,
    order = 1,
    seasonal = True,
    drop = True,
    additional_terms = [fourier]
)
X_time = dp.in_sample()
X_time['NewYearsDay'] = (X_time.index.dayofyear == 1)

model = LinearRegression(fit_intercept = False)
model.fit(X_time, y)
y_deseason = y - model.predict(X_time)
y_deseason.name = 'sales_deseasoned'

ax = y_deseason.plot()
ax.set_title("Magazine Sales (deseasonalized)")
```

![230505-FV2](https://user-images.githubusercontent.com/130429032/236466094-6f6d3905-9ab0-4ed3-a61a-5ed2f633a3d7.png)

- 위 코드는 시계열 데이터의 계절성을 제거한 후 그래프를 그리는 코드입니다.
- **‘.squeeze()’** 메소드는 차원 중 크기가 1인 차원을 제거하여 데이터를 다시 배열합니다. 만약 차원 중 크기가 1이 아닌 차원이 있다면 그대로 유지합니다.
- **‘CalendarFourier()’** 함수를 사용하여 월별 계절성을 모델링하는 **‘fourier’** 객체를 생성하고 이를 **‘additional_terms’** 인자로 **‘DeterministicProcess()’** 함수에 전달하여 **‘dp’** 객체를 생성합니다.
- **‘DeterministicProcess()’** 함수를 사용하여 시간 추세 모델링에 필요한 결정론적 요소를 추가합니다.
    - **‘constant = True’** : 모델에 절편을 추가합니다.
    - **‘index = y.index’** : 모델링에 사용할 인덱스를 지정합니다.
    - **‘order = 1’** : 추세를 모델링하기 위해 차분의 차수를 1로 설정합니다.
    - **‘seasonal = True’** : 계절성 패턴을 모델링하기 위해 계절성 구성 요소를 추가합니다.
    - **‘drop = True’** : 누락된 값이 있는 행을 제거합니다.
    - **‘additional_terms = [fourier]’** : fourier 변환을 통해 계절성 패턴을 모델링하는 추가적인 구성 요소를 추가합니다.
- **‘dp.in_sample()’** 은 데이터프레임의 인덱스를 기반으로 결정론적(time series deterministic) 항을 생성합니다. **‘in_sample()’** 메소드는 데이터 프레임 인덱스를 기반으로 예측 변수 행렬을 반환합니다. 인덱스를 기반으로 한 다양한 계절성, 휴일 등의 정보를 사용하여 예측 변수를 생성합니다.
- **‘(X_time.index.dayofyear == 1)’** 는 Pandas DatetimeIndex에서 각각의 날짜에 대해 해당 날짜가 1월 1일인지 아닌지를 나타내는 boolean 배열을 반환합니다.

By lagging a time series, we can make its past values appear contemporaneous with the values we are trying to predict (in the same row, in other words). This makes lagged series useful as features for modeling serial dependence. To forecast series, we could use y_lag_1 and y_lag_2 as features to predict the target y.

시계열을 래깅(lagging)하여 과거 값을 현재 예측하려는 값과 동시에 나타나게 만들 수 있습니다. 이렇게 래깅된 시계열은 직렬 의존성 모델링에 유용한 특징(feature)으로 사용될 수 있습니다. 시계열을 예측하기 위해 y_lag_1 및 y_lag_2를 특징으로 사용하여 목표 y를 예측할 수 있습니다.

# 4.1 Lag plot

A lag plot of a time series shows its values plotted against its lags. Serial dependence in a time series will often become apparent by looking at a lag plot. The most commonly used measure of serial dependence is known as **autocorrelation**, which is simply the correlation a time series has with one of its lags. The **partial autocorrelation** tells you correlation of a lag accounting for all of the previous lags — the amount of “new” correlation the lag contributes, so to speak. Plotting the partial autocorrelation can help you choose which lag features to use.

시계열의 lag 플롯은 시계열 값이 그것의 lag에 대해 플롯된 것을 보여줍니다. 시계열의 직렬 의존성은 lag 플롯을 살펴봄으로써 종종 확인됩니다. 직렬 의존성의 가장 일반적으로 사용되는 측정치는 자기상관이라고 알려져 있으며, 이는 단순히 시계열이 자신의 lag 중 하나와 가지는 상관 관계입니다. 부분 자기상관은 이전 모든 lag을 고려한 lag의 상관 관계를 나타냅니다. 즉, lag이 기여하는 “새로운” 상관 관계의 양입니다. 부분 자기상관을 플롯하는 것은 어떤 lag features를 사용할지 선택하는 데 도움이 됩니다.

```python
def lagplot(x, y = None, lag = 1, standardize = False, ax = None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(alpha = 0.75, s = 3)
    line_kws = dict(color = 'C3', )
    ax = sns.regplot(x = x_, y = y_, scatter_kws = scatter_kws, line_kws = line_kws, lowess = True, ax = ax, **kwargs)
    at = AnchoredText(f"{corr:2f}", prop = dict(size = "large"), frameon = True, loc = "upper left")
    at.patch.set_boxstyle("square, pad = 0.0")
    ax.add_artist(at)
    ax.set(title = f"Lag {lag}", xlabel = x_.name, ylabel = y_.name)
    return ax
```

- **‘lagplot’** 함수는 주어진 시리즈의 지연된 값들(lagged values)과 해당 시리즈의 값들을 플로팅해주는 함수입니다. 플로팅된 그래프는 특정 시간 단위의 시리즈 값들과 이전 시점의 값들 간의 상관관계를 보여줍니다. 입력값은 다음과 같습니다.
    - **x** : 시리즈 데이터
    - **y** : y축에 대한 시리즈 데이터 (생략 가능)
    - **lag** : 지연(lag) 단위
    - **standardize** : 데이터의 표준화 여부
    - **ax** : 그래프를 표시할 matplotlib 축 (생략 가능)
    - ****kwargs** : seaborn regplot 함수에 대한 인수 (생략 가능)
- 이 함수는 다음과 같은 작업을 수행합니다. 시리즈 데이터를 lag 단위만큼 이동하여 지연된 값(x_)을 생성합니다. standardize가 True인 경우, 데이터를 표준화합니다. y가 주어진 경우, y도 standardize합니다. x_와 y_의 상관계수를 계산합니다. matplotlib과 seaborn을 사용하여 lag plot을 생성합니다. 그래프에 상관계수를 추가합니다. 그래프의 제목과 축 이름을 설정하고 그래프를 반환합니다.
- **‘x_ = x.shift(lag)’** 는 주어진 데이터 **‘x’** 를 **‘lag’** 시간 단위로 이동시키는 것을 의미합니다.
- **‘matplotlib.offsetbox.AnchoredText’** 는 그래프의 특정 위치에 텍스트를 추가하기 위한 기능을 제공합니다. 이 기능은 그래프에 주석을 추가하는 것과 유사하지만, 주석과 달리 위치가 고정되며 그래프가 이동하더라도 같은 위치에 텍스트가 유지됩니다. 일반적으로 그래프의 범례나 통계량을 추가하는 데 사용합니다.
    - **‘s’** : 표시할 문자열입니다.
    - **‘loc’** : 텍스트 박스를 배치할 위치입니다 (ex: ‘loc = ‘upper left’)
    - **‘prop’** : 텍스트 속성
    - **‘frameon’** : 박스 외각선 표시 여부 (True 또는 False)
    - **‘bbox_to_anchor’** : 박스의 위치를 좌표로 지정합니다. ‘loc’과 함께 사용됩니다.
    - **‘pad’** : 박스와 문자열 사이의 여백입니다.
    - **‘borderpad’** : 박스와 외곽과의 여백입니다.
- **‘dict(alpha = 0.75, s = 3)’** 는 **‘alpha’** 키워드를 0.75로, **‘s’** 키워드를 3으로 설정한 딕셔너리입니다. **‘alpha’** 는 산점도 점의 투명도를 나타내는 값이며, 0에서 1 사이의 실수값을 가집니다. **‘s’** 는 산점도 점의 크기를 나타내는 값이며 정수값을 가집니다.
- **‘at.patch.set_boxstyle(”square, pad = 0.0”** 는 AnchoredText의 패치에 박스 스타일을 설정하는 코드입니다. 패치는 텍스트의 배경 부분을 의미합니다. ‘**Square’**는 박스의 모양을 사각형으로 설정하고 **‘pad = 0.0’**은 박스와 텍스트 사이의 간격을 0으로 설정합니다. 이렇게 하면 텍스트가 박스에 꽉 차게 표시됩니다.
- **‘ax.add_artist()’**는 Axes 객체에 새로운 아티스트를 추가하는 메소드입니다. 그래프 내부의 특정 위치에 텍스트 박스를 추가하는 데 사용됩니다. 기존의 아티스트와 겹치지 않는 새로운 아티스트를 추가합니다.

```python
def plot_lags(x, y = None, lags = 6, nrows = 1, lagplot_kwargs = {}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2 + 10, nrows * 2 + 5))
    fig, axs = plt.subplots(sharex = True, sharey = True, squeeze = False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag = k + 1, ax = ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict = dict(fontsize = 14))
            ax.set(xlabel = "", ylabel = "")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel = x.name)
    plt.setp(axs[:, 0], ylabel = y.name if y is not None else x.name)
    fig.tight_layout(w_pad = 0.1, h_pad = 0.1)
    return fig
```

- **‘plot_lags’** 함수는 시계열 데이터의 여러 시차(lag)를 한 번에 시각화할 수 있게 도와주는 함수입니다.
- **‘kwargs’**는 **‘plot_lags()’** 함수에 전달되는 모든 키워드 인자들을 포함하는 딕셔너리입니다.
- **‘kwargs.setdefault(’nrows’, nrow)’**는 **‘nrows’**가 **‘kwargs’** 딕셔너리에 이미 존재하는 경우에는 **‘nrows’** 값을 반환하고, 존재하지 않는 경우 **‘nrows’** 키와 그 값을 추가합니다. 즉, **‘nrows’**가 따로 입력되지 않은 경우에는 **‘nrows’**의 기본값인 1로 설정됩니다.
- **‘math.ceil()’** 함수는 안의 내용보다 큰 최소의 정수를 반환합니다. 다시 말해 올림값을 구하는 함수입니다.
- **‘squeeze = False’** 를 설정하면, 반환된 AxesSubplot 객체가 2차원 배열 형태가 됩니다. 만약 **True**로 설정하면, 1차원 형태의 배열로 반환됩니다.
- **‘**kwargs’**는 키워드 인자를 받는 매개변수입니다. 이를 통해 사용자가 추가적으로 설정할 수 있는 인자들을 전달할 수 있습니다.

Let’s take a look at the **lag** and **autocorrelation plots** first:

```python
_ = plot_lags(y_deseason, lags = 8, nrows = 2)
```

![230505-FV3](https://user-images.githubusercontent.com/130429032/236466101-a16351bc-7779-4366-8436-656d4c240e90.png)

```python
from statsmodels.graphics.tsaplots import plot_pacf
_ = plot_pacf(y_deseason, lags = 8)
```

![230505-FV4](https://user-images.githubusercontent.com/130429032/236466106-fbd156df-b886-4f8a-8240-7cc6b0e9054d.png)

Here we examine leading and lagging values for **onpromotion** plotted againest **magazine sales**.

여기에선 잡지 판매량에 대해 플롯된 onpromotion의 선행 및 후행 값을 살펴봅니다.

```python
onpromotion = mag_sales.loc[:, 'onpromotion'].squeeze().rename('onpromotion')

# Drop the New Year outlier
plot_lags(x = onpromotion.iloc[1:], y = y_deseason.iloc[1:], lags = 3, nrows = 1)
```

![230505-FV5](https://user-images.githubusercontent.com/130429032/236466110-b887876f-a548-4554-b1fa-16771cf33810.png)

# 4.2 Lags. Forecasting

after that, we can make lags for future plots.

```python
def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}' : ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis = 1
    )
```

- 이 함수는 시계열 데이터와 최대 지연 수(lags)를 입력으로 받아서 해당 시계열 데이터에 대해 지연된 데이터를 생성하는 함수입니다.

```python
X = make_lags(y_deseason, lags = 4)
X = X.fillna(0.0)
```

- 위 코드는 **‘make_lags’** 함수를 이용해 시계열 데이터 **‘y_deseason’**의 지연(lag) 값을 구하고, **‘NaN’** 값을 0.0으로 채워서 **‘X’** 데이터프레임을 생성하는 코드입니다.
- 

```python
# Create target series and data splits
y = y_deseason.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 60, shuffle = False)

# Fit and predict
model = LinearRegression() # 'fit_intercept = True' since we didn't use DeterministicProcess
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index = y_train.index)
y_fore = pd.Series(model.predict(X_test), index = y_test.index)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 10))
ax = y_train.plot(color = "0.75", style = ".-", markeredgecolor = "0.25", markerfacecolor = "0.25", ax = ax)
ax = y_test.plot(color = "0.75", style = ".-", markeredgecolor = "0.25", markerfacecolor = "0.25", ax = ax)
ax = y_pred.plot(ax = ax)
ax = y_pred.plot(ax = ax)
_ = y_fore.plot(ax = ax, color = 'C3')
plt.show()
```

![230505-FV6](https://user-images.githubusercontent.com/130429032/236466111-3b6386fc-bf29-4f87-85ae-e4f14ef8552f.png)

- **‘y_deseason’**을 타겟 시계열로 복사합니다.
- **‘make_lags()’** 함수를 사용하여 **‘ydeseason’** 에 대한 **‘lags = 4’**의 지연 시계열을 만듭니다.
- **‘X_train’, ‘X_test’, ‘y_train’, ‘y_test’**로 데이터를 분할합니다. **‘test_size’**는 60으로 지정되어 있습니다.
- **‘LinearRegression()’** 모델을 만들고 **‘X_train’, ‘y_train’** 으로 모델을 학습합니다.
- **‘y_train’**의 인덱스를 사용하여 모델을 사용하여 **‘y_train’**에 대한 예측값 **‘y_pred’**를 생성합니다.
- **‘y_test’**의 인덱스를 사용하여 모델을 사용하여 **‘y_test’**에 대한 예측값 **‘y_fore’**를 생성합니다.
- **‘y_train’, ‘y_test’, ‘y_pred’, ‘y_fore’**를 그래프로 그려줍니다.
- 그래프는 **‘y_train’**과 **‘y_test’**는 회색으로, **‘y_pred’**는 파랑색으로, **‘y_fore’**는 빨간색으로 표시됩니다. 그래프는 타겟 시계열의 실제 값과 모델 예측 값을 시각적으로 비교할 수 있도록 표시됩니다.