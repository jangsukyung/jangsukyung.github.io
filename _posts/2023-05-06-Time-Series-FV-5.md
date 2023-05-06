---
layout: single
title:  "Store Sales. Time Series Forecast & Visualization 5"
categories: Transcription
tag: [Python, Kaggle, Store, Sales, Transcription]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

매장 매출. 시계열 예측 및 시각화

# 6. Machine learning forecasting

```python
# train data
store_sales = df_train.copy()
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017']
)
```

- 이 코드는 각 제품군(family)과 일자(date)별 평균 매출(sales)을 계산하여 2017년 해당하는 데이터만 모아서 family_sales 데이터프레임으로 만드는 것입니다.

```python
# test data
test = df_test.copy()
test['date'] = test.date.dt.to_period('D')
test = test.set_index(['store_nbr', 'family', 'date']).sort_index()
```

- 테스트 데이터를 처리하는 코드입니다.

```python
def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
         axis = 1
    )
```

- **이 함수는 time series 데이터를 다중 스텝(time steps) 예측을 위한 타겟 변수로 변환하는 함수**입니다. 생성된 타겟 변수는 모두 하나의 데이터프레임으로 합쳐서 반환됩니다.

```python
y = family_sales.loc[:, 'sales']

# make 4 lag features
X = make_lags(y, lags = 5).dropna()

# make multistep target
y = make_multistep_target(y, steps = 16).dropna()

y, X = y.align(X, join = 'inner', axis = 0)
```

- y는 16개의 멀티스텝 타겟 시계열 데이터를 가지고 있으며, X는 y의 5개의 lag feature 시계열 데이터를 가지고 있습니다. X와 y의 인덱스는 일치하도록 처리되어 있습니다.
- **make_lags() 는 주어진 시계열 데이터에 대해 지정된 시간 간격 만큼의 래그 값을 계산하는 함수**입니다.
- **make_multistep_target() 는 주어진 시계열 데이터에 대해 다중 단계(target) 예측 문제를 해결하기 위해 다음 스텝을 예측하기 위한 여러 개의 타겟을 생성하는 함수**입니다.  예를 들어, **stpes = 16 으로 설정되어 있다면, 생성된 타겟 데이터프레임은 현재 시점으로부터 16스텝 뒤의 시계열 데이터를 예측하기 위한 16개의 다른 열로 구성**됩니다. 이렇게 생성된 다중 단계 예측 문제를 해결하기 위해서는, **모델링에서 다중 출력 모델(multi-output model)을 사용하거나 각각의 단계를 개별적으로 예측**하는 것이 필요합니다.

```python
le = LabelEncoder()
X = (X
     .stack('family') # wide to long
     .reset_index('family') # convert index to column
     .assign(family = lambda x: le.fit_transform(x.family)) # label encode
)
y = y.stack('family') # wide to long

display(y)
```

- 위 코드는 LabelEncoder()를 사용하여 X 데이터의 family feature를 label encoding하는 과정입니다. 먼저 stack() 메소드를 사용하여 wide format의 X 데이터를 long format으로 변경합니다. 그 후 reset_index() 메소드를 사용하여 인덱스를 컬럼으로 변환합니다. 그리고 assign() 메소드와 람다 함수를 사용하여 family feature를 LabelEncoder()로 label encoding 합니다. 마지막으로 y 데이터도 stack() 메소드를 사용하여 long format으로 변경합니다.
- **long format으로 변경하는 이유**
    - **데이터 분석에 용이하다** : 한 열에는 하나의 변수만 포함되어 있으므로 분석할 때 피벗 테이블과 같이 사용하기 쉽고 직관적입니다.
    - **시각화에 용이하다** : 여러 가지 family에 대한 시계열 데이터를 같은 차트에 표시할 수 있습니다.
    - **머신 러닝 모델 적용에 용이하다** : 모델의 입력 변수와 대상 변수를 각각 따로 두기 쉽기 때문입니다. 이는 모델 학습의 안정성과 일반화 성능을 높일 수 있습니다.

# 6.1 Forecast with the DirRec strategy

DirRec 전략으로 예측

Instatiate a model that applies the DirRec strategy to XGBoost.

XGBoost에 DirRec 전략을 적용하는 모델을 인스턴스화합니다.

```python
# init model
model = RegressorChain(base_estimator=XGBRegressor())
```

- 이 코드는 XGBRegressor를 기반으로 하는 RegressorChain 모델을 초기화하는 것입니다. **RegressorChain 은 다중 출력 회귀(multi-output regression) 문데를 해결하는 데 사용되는 방법 중 하나**입니다. 이 방법은 **다른 출력 값에 대한 예측을 현재 출력 값에 대한 예측의 특성으로 모델링하여 다중 출력 문제를 단일 출력 문제로 변환**합니다.

```python
# train model
model.fit(X, y)
y_pred = pd.DataFrame(model.predict(X), index=y.index,columns=y.columns).clip(0.0)
```

- RegressorChain 모델을 이용하여 X와 y를 이용하여 학습을 수행합니다. 모델이 학습을 완료하면, 모델을 사용하여 예측값 y_pred를 생성합니다. 이 때 y_pred도 y와 동일한 형태를 가지며, clip 함수를 이용하여 예측값을 0.0 이상으로 제한합니다.

Also, we need to dfine helpfull function, **plot_multistep**:

```python
# helpful function
def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return ax
```

- 이 함수는 다중 시계열 예측 결과를 시각화하는데 도움을 주는 함수입니다. y 는 다중 시계열 예측 결과이며, every 는 몇 번째 예측 결과를 그릴 것인지를 나타내는 인자입니다. 예를 들어, every = 1 이면 모든 예측 결과를 그리고, every = 2 이면 2개씩 건너뛰며 그립니다. ax 는 그래프를 그릴 Matplotlib의 Axes 객체입니다. 만약 ax 가 주어지지 않으면 새로운 Axes 객체를 생성합니다. **pallete_kwargs 는 Seaborn 팔레트 생성에 사용할 인자를 나타내는 딕셔너리입니다.** 이 함수는 시계열을 다른 색으로 구분하여 하나의 그래프에 그리게 됩니다.

So, now, we can **plot results**:

```python
FAMILY = 'BEAUTY'
START = '2017-04-01'
EVERY = 16

y_pred_ = y_pred.xs(FAMILY, level='family', axis=0).loc[START:]
y_ = family_sales.loc[START:, 'sales'].loc[:, FAMILY]

fig, ax = plt.subplots(1, 1, figsize=(11, 4))
ax = y_.plot(color="0.75",style=".-",markeredgecolor="0.25", markerfacecolor="0.25",ax=ax, alpha=0.5)
ax = plot_multistep(y_pred_, ax = ax, every = EVERY)
_ = ax.legend([FAMILY, FAMILY + ' Forecast'])
```

![2023-05-06-2-](https://user-images.githubusercontent.com/130429032/236622536-1b16a270-6e0b-4c52-9ed5-038cdf9b5bb5.png)

- 위 코드는 BEAUTY 카테고리의 판매량과 예측 결과를 비교하는 그래프를 그리는 코드입니다.