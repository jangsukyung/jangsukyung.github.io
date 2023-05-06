---
layout: single
title:  "Store Sales. Time Series Forecast & Visualization 4"
categories: Practice
tag: [Python, Kaggle, Store, Sales, Practice]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

매장 매출. 시계열 예측 및 시각화

# 5. Hybrid Models

Linear regression excels at extrapolating trends, but can’t learn interactions.

XGBoost excels at learning interactions, but can’t extrapolate trends. Here we’ll learn how to create “hybrid” forecasters that combine complementary learning algorithms and let the strengths of one make up for the weakness of the other.

선형 회귀는 추세를 추정하는 데 탁월하지만 상호작용을 학습할 수 없습니다.

XGBoost는 상호작용을 학습하는 데 뛰어나지만 추세를 추정할 수 없습니다. 여기에서는 서로 보완적인 학습 알고리즘을 결합하여 “하이브리드” 예측 모형을 만드는 방법을 배우게 됩니다. 이 방법은 한 알고리즘의 강점이 다른 알고리즘의 약점을 보완하도록 하여 보다 정확한 예측 결과를 도출할 수 있도록 합니다.

```python
store_sales = df_train.copy()
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family') # 인덱스 중 하나를 컬럼으로 변
    .loc['2017']
)
```

- **‘family_sales’** 데이터프레임은 2017년에 대해 각 제품군(family)별 평균 매출을 나타내는 데이터프레임입니다.

Firstly, we should create **Boosted Hybrid class**:

```python
# we'll add fit and predict methods to this minimal class
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None # store column names from fit method
```

- 위 코드는 **BoostedHybrid** 클래스를 정의하는 코드입니다. BoostedHybrid 클래스는 **model_1**과 **model_2** 두 가지 모델을 조합하여 예측 모델을 생성하는 클래스입니다.
- 클래스에는 **init()** 메소드가 정의되어 있으며, 이 메소드는 인스턴스를 생성할 때 호출됩니다. 메소드 내부에서는 **self.model_1**과 **self.model_2** 변수에 **model_1**과 **model_2** 인자를 할당하고, self_.y_columns 변수를 None으로 초기화합니다. 이 변수는 **fit()** 메소드에서 사용할 예정입니다**.**
- **BoostedHybrid** 클래스에는 두 개의 메소드가 추가될 예정입니다. **fit()** 메소드와 **predict()** 메소드가 추가될 예정이며, 이 두 메소드를 통해 학습하고 예측을 수행할 수 있습니다.
- Python에서 클래스를 정의할 때, 클래스가 생성될 때 자동으로 호출되는 특별한 메소드가 있는데 이 메소드를 클래스의 **생성자(constructor)**라고 하며, **init()**이라는 이름으로 정의됩니다.
- **init()** 메소드는 클래스의 인스턴스를 초기화하고 속성 값을 설정하기 위해 사용됩니다. 클래스를 생성할 때 인스턴스 변수를 초기화하거나, 인자를 받아서 속성 값을 설정하는 등의 작업을 수행할 수 있습니다.

Also, we need to create **fit** method:

```python
def fit(self, X_1, X_2, y):
    # train model_1
    self.model_1.fit(X_1, y)

    # make_predictions
    y_fit = pd.DataFrame(
        self.model_1.predict(X_1),
        index = X_1.index, columns = y_columns
    )

    # compute residuals
    y_resid = y - y_fit
    y_resid = y_resid.stack().squeeze() # wide to long

    # train model_2 on residuals
    self.model_2.fit(X_2, y_resid)

    # save column names for predict method
    self.y_columns = y.columns
    # Save data for queistion checking
    self.y_fit = y_fit
    self.y_resid = y_resid

# Add method to class
BoostedHybrid.fit = fit
```

- 위 코드는 BoostedHybrid 클래스에 **fit()** 메소드를 추가하는 코드입니다.
- **fit()** 메소드는 두개의 인자 X_1, X_2와 목표 변수 y를 받습니다. 메소드 내부에서는 먼저 model_1을  X_1과 y를 이용하여 학습시킵니다. 이후 model_1을 이용하여 X_1 데이터에 대한 예측값을 계산합니다.
- 그 다음으로 **모델1의 예측값과 실제 목표 변수와의 잔차(residual)를 계산합니다.** 잔차는 실제 목표 변수에서 모델1의 예측값을 뺀 값입니다. 이 잔차는 model_2를 학습시키는 데 사용됩니다.
- 마지막으로 **model_2를 X_2와 잔차 데이터를 이용하여 학습시킵니다.** 이후 y_columns 변수를 목표 변수 y의 컬럼 이름으로 설정하고, y_fit과 y_resid 변수에는 fit() 메소드에서 계산된 예측값과 잔차를 저장합니다.
- BoostedHybrid.fit() 메소드는 이후 BoostedHybrid 클래스에 추가되어서 BoostedHybrid 객체에서 호출할 수 있습니다.

And **predict** method:

```python
def predict(self, X_1, X_2):
    # Predict with model_1
    y_pred = pd.DataFrame(
        self.model_1.predict(X_1),
        index = X_1.index, columns = self.y_columns,
    )
    y_pred = y_pred.stack().squeeze() # wide to long

    # Add model_2 predictions to model_1 predictions
    y_pred += self.model_2.predict(X_2)

    return y_pred.unstack()

# Add method to class
BoostedHybrid.predict = predict
```

- 위 코드는 BoostedHybrid 클래스에 **predict()** 메소드를 추가하는 함수입니다.
- predict() 메소드는 두 개의 인자 X_1과 X_2를 받습니다. **메소드 내부에서는 먼저 model_1을 X_1을 이용하여 예측합니다.** 그리고 이 예측값을 y_pred에 저장합니다.
- 그 다음으로는 **model_2를 X_2를 이용하여 예측하고, y_pred에 model_2의 예측값을 더합니다.** 이후 y_pred를 다시 wide-format으로 변환하여 반환합니다.
- BoostedHybrid.predict() 메소드는 이후 BoostedHybrid 클래스에 추가되어서 BoostedHybrid 객체에서 호출할 수 있습니다. **예측값은 학습된 모델에 따라서 계산되며, 이전에 fit() 메소드를 통해 계산된 예측값과 잔차가 함께 사용됩니다.**
- **stack()** 메소드를 사용하면 데이터프레임을 **‘long-format’으로 변경**할 수 있습니다. 이렇게 변환된 데이터는 인덱스와 열 이름을 각각의 열로 가지며, 데이터 값은 단일 열로 구성됩니다.
- **squeeze()** 메소드는 단일 열을 가진 데이터프레임에서 시리즈를 추출합니다. 이를 통해 y_pred는 데이터프레임에서 시리즈 형태로 변환됩니다.

Here we can set up data for training:

```python
# Target series
y = family_sales.loc[:, 'sales']

# X_1: Features for Linear Regression
dp = DeterministicProcess(index = y.index, order = 1)
X_1 = dp.in_sample()

# X_2: Features for XGBoost
X_2 = family_sales.drop('sales', axis = 1).stack() # onpromotion feature

# Label encoding for 'family'
le = LabelEncoder() # from sklearn.preprocessing
X_2 = X_2.reset_index('family')
X_2['family'] = le.fit_transform(X_2['family'])

# Label encoding for seasonality
X_2["day"] = X_2.index.day # values are day of the month
```

- 위 코드는 BoostedHybrid 모델에 적용할 수 있는 입력 데이터를 준비하는 과정입니다.
    - y : 예측하려는 대상 시계열 데이터인 sales 컬럼을 선택합니다.
    - X_1 : 선형 회귀 모델에 사용할 시간 정보(’dp’)를 생성합니다. **dp.in_sample() 메소드를 사용하여 대상 시계열과 동일한 인덱스를 가지는 1차 차분값을 생성합니다.**
    - X_2 : sales 컬럼을 제외한 모든 변수를 이용하여, XGBoost 모델에 사용할 수 있는 형태로 데이터를 변환합니다. **stack() 메소드를 이용하여 wide-format 데이터프레임을 long-format으로 변경합니다.**
    - le : family 변수에 대해 label encoding을 수행합니다. **인덱스에 있는 family 변수를 데이터프레임의 열로 가져오고, LabelEncoder() 를 이용하여 범주형 변수를 정수형으로 인코딩합니다.**
    - X_2[”day”] = X_2.index.day : X_2 데이터프레임의 인덱스에서 day 정보만 추출하여, **day 변수를 추가합니다. 이 변수는 시계열 데이터의 계절성을 고려하기 위한 정보입니다.** 예를 들어 월말과 월초의 판매량은 큰 차이를 보일 수 있으므로, 이에 대한 정보를 반영하기 위해 day 변수를 추가합니다.

Here we can train our model:

```python
# Create model
model = BoostedHybrid(
    model_1 = LinearRegression(),
    model_2 = XGBRegressor()
)

model.fit(X_1, X_2, y)

y_pred = model.predict(X_1, X_2)
y_pred = y_pred.clip(0,0)
```

- BoostedHybrid 클래스로부터 모델 객체(model)를 생성합니다. 생성된 모델은 Linear Regression 모델(model_1)과 XGBoost 모델(model_2)를 조합한 하이브리드 모델입니다.
- 그 후, fit 메소드를 이용하여 모델을 학습니다. fit 메소드의 입력값으로는 Linear Regression 모델에 사용할 feature인 X_1, XGBoost 모델에 사용할 feature인 X_2,  그리고 target series인 y가 들어갑니다. **이를 이용하여 먼저 model_1인 Linear Regression 모델을 학습시키고, 이 모델을 이용하여 y값의 예측치를 계산합니다. 이를 이용하여 model_2인 XGBoost 모델에 대한 학습을 진행합니다. 그리고 예측 결과를 저장합니다.**
- 마지막으로 predict 메소드를 이용하여 예측 결과를 얻습니다. 입력값으로는 X_1(feature for Linear Regression)과 X_2(feature for XGBoost)가 들어갑니다. **이를 이용하여 먼저 model_1인 Linear Regression 모델을 이용하여 y의 예측치를 계산합니다. 이후 model_2인 XGBoost 모델의 예측 결과를 더하여 최종 예측 결과를 얻습니다. 최종 예측 결과에 대해 clip 메소드를 이용하여 예측 결과가 0 이하인 경우 0으로 반환합니다.**

After that we train and plot

```python
y_train,y_valid =y[:"2017-07-01"],y["2017-07-02":]
X1_train,X1_valid =X_1[: "2017-07-01"],X_1["2017-07-02" :]
X2_train,X2_valid =X_2.loc[:"2017-07-01"],X_2.loc["2017-07-02":]

# Some of the algorithms above do best with certain kinds of
# preprocessing on the features (like standardization), but this is
# just a demo.
model.fit(X1_train,X2_train,y_train)
y_fit =model.predict(X1_train,X2_train).clip(0.0)
y_pred =model.predict(X1_valid,X2_valid).clip(0.0)

families =y.columns[0:6]
axs =y.loc(axis=1)[families].plot(subplots=True,
                                   sharex=True,
                                   figsize=(30, 20),
                                   color="0.75",
                                   style=".-",
                                   markeredgecolor="0.25",
                                   markerfacecolor="0.25",
                                   alpha=0.5)
_ =y_fit.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C0', ax=axs)
_ =y_pred.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C3', ax=axs)
for ax,family in zip(axs,families):
		ax.legend([]) # 범례 제
		ax.set_ylabel(family)
```

![2023-05-06-1-](https://user-images.githubusercontent.com/130429032/236619458-db302879-5aa6-42c8-9365-e9aea76ba997.png)

- 위 코드는 train 데이터와 validation 데이터에 대해 model.fit() 과 model.predict() 를 수행하고, 그 결과를 시각화하는 코드입니다.