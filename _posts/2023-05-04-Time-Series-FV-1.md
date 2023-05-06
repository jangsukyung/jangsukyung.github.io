---
layout: single
title:  "Store Sales. Time Series Forecast & Visualization 1"
categories: Transcription
tag: [Python, Kaggle, Store, Sales, Transcription]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

매장 매출. 시계열 예측 및 시각화

# 1. Import libraries

```python
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from pandas import date_range
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
```

- **import math**: 파이썬 math 모듈을 불러옵니다. math 모듈은 다양한 수학 함수와 상수를 제공합니다.
- **import numpy as np**: NumPy는 수치 계산을 위한 파이썬 라이브러리로, 다차원 배열을 다루는 기능 등을 제공합니다.
- **import pandas as pd**: 데이터 조작과 분석을 위한 라이브러리로, 표 형태의 데이터를 다루는 기능 등을 제공합니다.
- **import seaborn as sns**: matplotlib을 기반으로 한 데이터 시각화 라이브러리로, 다양한 그래프 스타일과 테마를 제공합니다.
- **import matplotlib.pyplot as plt**: 데이터 시각화 라이브러리로 그래프를 그리는 기능 등을 제공합니다.
- **from matplotlib.colors import ListedColormap**: 이산적인 색상 맵을 생성하기 위한 클래스입니다.
- **from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess**: 시계열 분석을 위한 클래스입니다.
- **from sklearn.linear_model import LinearRegression**: 선형 회귀 모델을 구현하는데 사용됩니다.
- **from pandas import date_range**: date_range 함수는 일정한 간격으로 날짜 / 시간을 생성하는 데 사용됩니다.
- **from statsmodels.graphics.tsaplots import plot_pacf**: PACF(Partial Autocorrelation Function)을 시각화하는 함수입니다.
- **from sklearn.model_selection import train_test_split**: train_test_split 함수는 데이터를 학습용과 검증용으로 분할하는 데 사용됩니다.
- **from sklearn.preprocessing import LabelEncoder**: 범주형 변수를 수치형 변수로 변환하는 데 사용됩니다.
- **from xgboost import XGBRegressor**: 회귀 모델링을 구현하는 데 사용됩니다.

```python
# Model 1(trend)
from sklearn.linear_model import ElasticNet, Lasso, Ridge

# Model 2
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import RegressorChain
import warnings
```

- **from sklearn.linear_model import ElasticNet, Lasso, Ridge**: 이 클래스들은 모두 선형 회귀 모델링에 사용되는 클래스로, Ridge와 Lasso는 규제(regularization)를 적용하는 선형 회귀 모델이며, ElasticNet은 L1 규제와 L2 규제를 모두 적용하는 선형 회귀 모델입니다.
- **from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor**: 모두 앙상블(ensemble) 기법을 이용한 회귀 모델링에 사용되는 클래스입니다. 랜덤 포레스트 (Random Forest) 알고리즘을 이용한 회귀 모델링을 수행합니다.
- **from sklearn.neightbors import KNeighborsRegressor**: 최근접 이웃(K-Nearest Neighbor) 알고리즘을 이용한 회귀 모델링에 사용되는 클래스입니다.
- **from sklearn.neural_network import MLPRegressor**: 다층 퍼셉트론(Multi-Layer Perceptron) 알고리즘을 이용한 회귀 모델링에 사용되는 클래스입니다.
- **from sklearn. multioutput import RegressorChain**: 다중 출력 회귀(Multi-Output Regression) 모델링에 사용되는 클래스로, 체인 방법(chain method)을 이용하여 다중 출력 회귀 모델을 구축합니다.
- **import warnings**: 경고 메시지를 표시하거나 제어하는 데 사용되는 모듈을 불러오는 코드입니다.

```python
# switch off the warnings
warnings.filterwarnings("ignore")
```

- 경고 메시지를 무시하도록 설정하는 함수입니다. 이를 호출하면 경고 메시지가 더 이상 표시되지 않습니다.

# 2. Read data

```python
df_holidays = pd.read_csv('holidays_events.csv', encoding = 'cp949', header = 0)
df_oil = pd.read_csv("oil.csv", encoding = 'cp949', header = 0)
df_stores = pd.read_csv("stores.csv", encoding = 'cp949', header = 0)
df_trans = pd.read_csv("transactions.csv", encoding = 'cp949', header = 0)

df_train = pd.read_csv("train.csv", encoding = "cp949", header = 0)
df_test = pd.read_csv("test.csv", encoding = "cp949", header = 0)
```

- ************‘header = 0’************ 은 CSV 파일의 첫 번째 줄이 열 이름을 포함하고 있다는 것을 나타내며, 이를 데이터프레임의 열 이름으로 사용하겠다는 의미입니다.

Also, we need to convert all ‘date’ columns to datetime Pandas format:

또한 우리는 모든 ‘date’ 열을 Pandas 형식의 날짜/시간 형식으로 변환해야 합니다:

```python
df_holidays['date'] = pd.to_datetime(df_holidays['date'], format = "%Y-%m-%d")
df_oil['date'] = pd.to_datetime(df_oil['date'], format = "%Y-%m-%d")
df_trans['date'] = pd.to_datetime(df_trans['date'], format = "%Y-%m-%d")
df_train['date'] = pd.to_datetime(df_train['date'], format = "%Y-%m-%d")
df_test['date'] = pd.to_datetime(df_test['date'], format = "%Y-%m-%d")
```

After that, We can look and check our different dataframes:

그런 다음 우리는 다양한 데이터 프레임을 살펴보고 확인할 수 있습니다.

Here we can see **df_holidays**:

여기서 우리는 df_holidays를 볼 수 있습니다.

```python
df_holidays.head(10) # check data
```

![time-series-fv-1-](https://user-images.githubusercontent.com/130429032/236100045-3d7f9248-05c3-4532-8ed8-1252c3f8859c.png)

Here we can see **df_oil**:

```python
df_oil.head(3) # check data
```

![time-series-fv-1-1](https://user-images.githubusercontent.com/130429032/236100046-aad19c3e-26fc-471d-8b36-7ec096393372.png)

Here we can see **df_stores**:

```python
df_stores.head(10) # check data
```

![time-series-fv-1-2](https://user-images.githubusercontent.com/130429032/236100052-884125ec-df8b-468c-920c-77bbe603dabd.png)

Here we can see **df_train**:

```python
df_train.head(10) # check data
```

![time-series-fv-1-3](https://user-images.githubusercontent.com/130429032/236100057-8343dbab-7c08-4d87-ae70-1c59ea23acd3.png)

Here we can see **df_test**:

```python
df_test.head(5) # check data
```

![time-series-fv-1-4](https://user-images.githubusercontent.com/130429032/236100061-ea2a71ce-0867-4098-8aea-f4ea237adf70.png)