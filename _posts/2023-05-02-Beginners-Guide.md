# Exploring TIme Series plots: Beginners Guide

시계열 플롯 탐색: 초보자 가이드

This notebook is an introductory notebook for visualizing and understanding different plots in time series data.

이 노트북은 시계열 데이터의 다양한 그림을 시각화하고 이해하기 위한 입문 노트북입니다.

## Imports

```python
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 12
time = np.arrange(5 * 365 + 1) # 5 years
```

- np.arange() 함수는 일정한 간격으로 배열을 생성하는 함수입니다. 하루 간격으로 5년의 시간을 나타냅니다.
- 이 코드는 5년간의 일일 시간 데이터를 생성하는 것입니다.

# Trend

```python
def plot_series(time, series, format = "-", start = 0, end = None, label = None, color = None):
	plt.plot(time[start:end], series[start:end], format, label = label, color = color)
	plt.xlabel("Time")
	plt.ylabel("Value")
	if label:
		plt.legend(fontsize = 14)
	plt.grid(True)
```

- 이 코드는 시간 데이터(time)와 해당 시간에 따른 값을 나타내는 데이터(series)를 입력받아 그래프를 출력하는 함수입니다.
- plt.plot()함수를 사용해 그래프를 그리고, 함수 인자로 받은 format, label, color 등을 설정합니다.
- start와 end 인자로 지정된 범위 내의 데이터만 그래프로 출력하며, xlabel과 ylabel 함수를 사용하여 x, y축의 레이블을 설정합니다.
- label 인자가 입력된 경우, plt.legend()함수를 사용하여 범례를 생성합니다.
- plt.grid()함수를 사용하여 그래프에 격자를 추가합니다.

```python
def trend(time, slope = 0):
	return slope * time
```

- 이 코드는 시간 데이터(time)와 경사도(slope)를 입력으로 받아 해당 시간에 대한 경사도 값을 계산하는 함수입니다.
- trend 함수는 입력된 경사도(slope)와 시간(time)을 곱한 결과를 반환합니다.
- 이 함수는 시계열 데이터의 경향성(trend)을 모델링하기 위해 사용될 수 있습니다.
- 경사도가 양수인 경우, 시간이 지남에 따라 값이 증가하는 경향을 나타내고, 음수인 경우에는 값이 감소하는 경향을 나타냅니다.
- 경사도가 0이면, 시간에 따른 값의 변화가 없는 경우를 나타냅니다.

## Trend Plot 1

```python
slope = 0.1
series = trend(time, slope)
plt.figure(figsize = (20, 6))
plot_series(Time, series, color = "purple")
plt.title("Trend Plot - 1", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide](https://user-images.githubusercontent.com/130429032/235585118-9ad6d07c-d9bf-44d7-bc50-a75bb95d37a7.png)

- 이 코드는 시간(time)과 경사도(slope)값을 이용하여 경향성(trend)이 있는 데이터를 생성하고 시각화합니다.
- 경사도(slope)는 0.1로 설정되어 있습니다. 그 다음으로, trend 함수를 이용하여 시간(time)에 대한 경향성이 포함된 데이터를 생성하고 이를 series 변수에 저장합니다.

## Trend Plot 2

```python
slope = -1
series = trend(time, slope)
plt.figure(figsize = (20, 6))
plot_series(time, series, color = "purple")
plt.title("Trend Plot - 2", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide1](https://user-images.githubusercontent.com/130429032/235585120-f850617a-358d-4bde-bb6e-44c7c3d02a70.png)

- 위 코드는 slope를 음수(-1)로 설정하여 time에 대한 시간의 경과에 따른 series의 값이 감소하는 선형적인 추세(trend)를 생성하는 코드입니다.
- slope가 음수로 설정되어 있기 때문에 시간의 경과에 따라 series의 값이 감소하는 추세가 나타납니다.

# Seasonality

```python
def seasonal_pattern(season_time):
	return np.where(season_time < 0.45,
									np.cos(season_time * 2 * np.pi),
									1 / np.exp(3 * season_time))
```

- seasonal_pattern은 계절성을 생성하기 위해 사용되는 함수입니다. np.where() 함수를 사용하여 season_time 값이 0.45보다 작으면 코사인 함수에 따라 계절성 값을 생성하고, 그렇지 않으면 지수 함수에 따라 계절성 값을 생성합니다.
- season_time : 0과 1 사이의 값을 가지는 시간 값으로, 계절성이 변하는 주기를 나타냅니다. 계절성이 1년 주기를 가진다면, ‘season_time’ 값은 0부터 1까지 1년 간의 시간을 나타내며, 값이 커질수록 해당 계절의 끝에 가까워집니다.
- np.cos(season_time * 2 * np.pi) : ‘season_time’값이 0.45보다 작을 경우, 0과 1 사이의 값을 가지는 코사인 함수를 통해 계절성 값을 생성합니다. 이 때, ‘np.pi’는 파이값을 나타내며, ‘2 * np.pi’는 360도(한 주기)를 라디안 값으로 나타낸 것입니다. 따라서, ‘season_time’값이 0이라면 코사인 함수 값은 1이 되고, ‘season_Time’값이 0.45에 가까워질수록 0으로 수렴합니다.
- ‘1 / np.exp(3 * season_time) : ‘season_time’ 값이 0.45보다 큰 경우, 1보다 작은 값을 가지는 지수 함수를 통해 계절성 값을 생성합니다. 이 때, ‘np.exp()’ 함수는 지수 함수(e^x)를 계산하는 함수입니다. 따라서 ‘season_time’값이 0이라면 계절성 값은 1이 되고, 1에 가까워질수록 0으로 수렴합니다.

```python
def seasonality(time, period, amplitude = 1, phase = 0):
	season_time = ((time + phase) % period) / period
	return amplitude * seasonal_pattern(season_time)
```

- 이 함수는 계절성(seasonality)을 나타내기 위해 주어진 기간(period)과 진폭(amplitude)으로 이루어진 주기 함수를 생성하는 함수입니다.
- ‘time’은 시간을 나타내는 데이터, ‘period’는 계절성 주기를 나타냅니다. 예를 들어, ‘period’가 365라면 1년 주기를 가지는 계절성을 의미합니다.
- ‘phase’는 시간의 시작점을 나타내는데, 주어진 ‘time’에서 ‘phase’값을 더한 후 ‘period’로 나누어 계절성 패턴이 반복되는 위치를 계산합니다. 예를 들어, ‘phase’가 0이면 계절성 패턴은 ‘time’의 시작점에서부터 시작합니다. ‘phase’가 ‘period’보다 작은 값이면 계절성 패턴은 ‘time’시작점보다 앞으로 이동하며, ‘phase’가 ‘period’보다 큰 값이면 패턴은 ‘time’ 시작점보다 뒤로 이동합니다.
- ‘season_time’은 주기적으로 반복된느 값을 생성하기 위해 계산된 변수입니다. ‘time’에서 ‘phase’를 더한 값의 ‘period’로 나누어서 0부터 1사이의 값으로 만듭니다. 이 값을 ‘seasonal_patter()’ 함수에 넣어서 계절성 패턴을 생성합니다.

## Seasonality Plot 1

```python
amplitude = 40
series = seasonality(time, period = 365, amplitude = amplitude, phase = 0)
plt.figure(figsize = (20, 6))
plot_series(time, series, color = "green")
plt.title("Seasonality Plot - 1"), fontdict = {'fontsize' : 20})
ptl.show()
```

![Beginners-Guide2](https://user-images.githubusercontent.com/130429032/235585121-09f58a45-7d87-42b1-872b-381edd9ed767.png)

- 위 코드는 연간 주기(365일)을 가지는 계절성을 생성하고, 시간에 따라 주기적으로 변화하는 값을 계산하여 그래프로 시각화합니다.
- ‘seasonality’ 함수는 입력된 ‘time’ 데이터를 연간 주기(365일)로 나누어 정규화한 값을 계산합니다. ‘amplitude’은 계절성 변화의 강도를 조절하며, ‘phase’는 계절성 그래프의 이동을 조절합니다.
- 이렇게 계산된 계절성 값을 ‘series’ 변수에 저장하고, ‘plot_series’함수를 이용하여 시계열 데이터를 그래프로 출력합니다.

## Seasonality Plot 2

```python
amplitude = 100
series = seasonality(time, period = 90, amplitude = amplitude, phase = 25)
plt.figure(figsize = (20, 6))
plot_series(time, series, color = "green")
plt.title("Seasonality Plot - 2", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide3](https://user-images.githubusercontent.com/130429032/235585122-ac00f3cd-2641-4149-bad4-7df2c45c4050.png)

- 90일 주기를 갖는 계절성 변화를 보여주며, 진폭은 100으로 설정되었고, 시작 지점은 25일로 설정되었습니다.

## Seasonality + Trend

Combined plot for seasonality and trend together

```python
baseline = 10 # 기준선
slope = 0.08 # 경사도
amplitude = 40 # 진폭
series = baseline + trend(time, slope) + seasonality(time, period = 365, amplitude = amplitude)
plt.figure(figsize = (20, 6))
plot_series(time, series, color = "green")
plt.title("Seasonality + Trend Plot", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide4](https://user-images.githubusercontent.com/130429032/235585124-65221bf0-e2ba-49d2-9ee1-e9eeffed8b2c.png)

- 위 코드는 baseline(기준선), slope(경사도), amplitude(진폭)을 설정하여 시간축에 대한 시계열 데이터(series)를 생성하고 이를 시각화한 코드입니다.
- ‘baseline’은 시계열 데이터의 기준선을 나타냅니다. 즉, 데이터가 기준선을 중심으로 어떻게 분포되는지를 나타냅니다.
- ‘slope’은 시계열데이터가 시간에 따라 어떻게 증가 또는 감소하는지를 나타냅니다.
- ‘amplitude’은 주기성을 갖는 시계열 데이터의 진폭을 나타냅니다. ‘seasonality’ 함수를 이용해 특정 주기를 가지는 패턴을 생성하고 이에 진폭을 곱해 시계열 데이터를 생성합니다.
- ‘series’는 기준선, 경사도, 주기성이 포함된 시계열 데이터를 나타냅니다.

# Noise

```python
def white_noise(time, noise_level = 1, seed = None):
	random = np.random.RandomState(seed)
	return random.random(len(time)) * noise_level
```

- 주어진 시간 범위에서 주어진 크기의 백색 잡음을 생성합니다. 랜던 시드 값이 주어지면 결과는 재현 가능합니다.
- ‘np.random.RandomState’ 클래스를 사용하여 랜덤 시드 값을 설정하고, ‘random.random’ 메소드를 사용하여 시간 범위와 동일한 길이의 랜덤한 값을 생성합니다. 생성된 값은 레벨 값으로 곱하여 레벨이나 크기를 조절합니다. 최종 결과는 시간과 함께 반환됩니다.

## Noise Plot

```python
noise_level = 10
noise = white_noise(time, noise_level, seed = RANDOM_SEED)
plt.figure(figsize = (20, 6))
plot_series(time[:200], noise[:200], color = "blue")
plt.title("Noise Plot", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide5](https://user-images.githubusercontent.com/130429032/235585126-79b148ca-4021-41e2-8f0a-1f3e7cedc5b4.png)

- 이 코드는 time 범위 내에 랜덤하게 생성된 백색 잡음을 반환하는 함수 white_noise()을 정의하고 있습니다. 함수는 입력값으로 time, noise_level, seed를 받습니다.
- time은 시간을 의미하는 값으로, np.arange()를 사용하여 5년 간의 값을 생성하고 있습니다.
- noise_level은 잡음의 크기를 조절하는 인자로, 기본값은 1입니다.
- seed는 재현성을 위한 램덤 시드 값으로, 기본값은 None이며, seed값이 지정되지 않으면 랜덤값이 사용됩니다.
    - 재현성 : 실험에서 관측된 사상을 같은 조건하에서 재실험하여 확실하게 할 수 있을 때, 그 실험은 재현성이 있다고 한다.
- 이 코드에서는 seed 값으로 RANDOM_SEED = 12를 사용하여 백색 잡음을 생성하고 있습니다. 생성된 잡음은 plot_series()함수를 사용하여 그래프로 시각화하고 있습니다. 200개의 데이터 포인트만 시각화되도록 slicing을 적용하고 있습니다.

## Noise + Seasonality + Trend

Combined plot for noise, seasonality and trend together

```python
series = baseline + trend(time, slope) + seasonality(time, period = 365, amplitude = amplitude)
series += white_noise(time, noise_level = 10, seed = RANDOM_SEED)
plt.figure(Figsize = (20, 6))
plot_series(time, series, color = "blue")
plt.title("Noise + Seasonality + Trend Plot", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide6](https://user-images.githubusercontent.com/130429032/235585127-28819006-90d7-4e64-9c27-7e430b0ac8b9.png)

- 이 코드는 baseline, trend, seasonality, white noise을 생성하고 모두 합쳐서 시계열 데이터(series)를 만드는 코드입니다.

# Autocorrelation

```python
def autocorrelation_1(time, amplitude, seed = None):
	rnd = np.random.RandomState(seed)
	a1 = 0.5
	a2 = -0.1
	rnd_ar = rnd.randn(len(time) + 50)
	rnd_ar[:50] = 100
	for step in range(50, len(time) + 50)
		rnd_ar[step] += a1 * rnd_ar[step - 50]
		rnd_ar[step] += a2 * rnd_ar[step - 33]
	return rnd_ar[50:] * amplitude
```

- 이 함수는 AR(2) 모형에 따라 시간에 따른 자기상관관계를 가지는 데이터를 생성하는 함수입니다.
- AR(2) 모형은 두 개의 이전 시간 스텝의 데이터와 현재 시간 스텝의 잔차(residual)의 선형 조합으로 현재 시간 스텝의 값을 예측하는 모델입니다.
- 이 함수에선 먼저 시간 스텝에 랜덤 잡음을 생성하고, 처음 50개의 값은 모두 100으로 설정합니다. 그리고 이후의 값들은 이전 50개의 값과 이전 33개의 값의 선형 조합에 랜덤 잡음을 더한 값으로 계산됩니다.
- 이렇게 계산된 시계열 데이터는 ‘rnd_ar[50:]에 저장되어 반환합니다. 반환된 값은 ‘amplitude’에 곱해진 후 반환됩니다.

```python
def autocorrelation_2(time, amplitude, seed = None):
	rnd = np.random.RandomState(seed)
	a1 = 0.8
	ar = rnd.randn(len(time) + 1)
	for step in range(1, len(time) + 1):
		ar[step] += a1 * ar[step - 1]
	return ar[1:] * amplitude
```

- 시간(time), 진폭(amplitude), 시드값(seed)를 받아서 주어진 시간 길이에 대한 자기상관이 있는(autocorrelated) 랜덤 시계열 데이터를 생성하는 함수입니다.
- 이 함수에서는 AR(1) 모형(autoregressive model of order 1)을 사용하여 랜덤 시계열 데이터를 생성합니다. AR(1)모형은 현재의 값이 이전 값에 의존하는 모형으로, 현재의 값은 이전 값에 일정한 비율(a1)만큼의 가중치를 더한 값으로 결정됩니다.
- 입력받은 시간 길이에 대해 랜덤 시계열 데이터를 생성한 후, AR(1) 모형을 적용하여 자기상관이 있는 데이터를 생성합니다. 이 때, 시드값을 설정하여 랜덤성을 제어할 수 있습니다.

## Autocorrelation Plot 1

```python
series = autocorrelation_1(time, amplitude, seed = RANDOM_SEED)
plt.figure(figsize = (20, 6))
plot_series(time[:200], series[:200], color = "red")
plt.title("Autocorrelation Plot - 1", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide7](https://user-images.githubusercontent.com/130429032/235585130-9d91e856-09c1-47e9-9eb9-ef90220cf6ee.png)

- 위 코드는 시계열 데이터에서 자기상관성(autocorrelation)을 가지는 시계열 데이터를 생성하는 함수를 사용해, 그 데이터를 시각화하는 코드입니다.
- ‘autocorrelation_1’ 함수는 주어진 시간대(time)에 따라, 시드(seed) 값에 의해 결정되는 무작위 노이즈 값에 대한 자기상관성을 가지는 시계열 데이터를 생성합니다. 이 때, 자기상관성 계수는 a1 = 0.5, a2 = -0.1 로 지정됩니다.

## Autocorrelation Plot 2

```python
series = autocorrelation_2(time, amplitude, seed = RANDOM_SEED)
plt.figure(figsize = (20, 6))
plot_series(time[:200], series[:200], color = "red")
plt.title("Autocorrelation Plot - 2", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide8](https://user-images.githubusercontent.com/130429032/235585131-9db9cf34-cfa7-4a39-a2b9-3d547301da1e.png)

- ‘autocorrelation_2()’ 함수는 강한 양의 자기상관관계를 가진 자료를 생성합니다. 이 함수는 랜덤 생성된 값에 이전 시점의 값에 일정한 비율만큼을 더해줌으로써 자기상관관계를 만듭니다.

## Autocorrelation + Trend

Combined plot for Autocorrelation and trend together

```python
amplitude = 10
slope = 2
series = autocorrelation_1(time, amplitude, seed = RANDOM_SEED) + trend(time, slope)
plt.figure(figsize = (20, 6))
plot_Series(time[:200], series[:200], color = "red")
plt.title("Autocorrelation + Trend Plot", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide9](https://user-images.githubusercontent.com/130429032/235585132-0a7a20f2-ce87-49b3-be0c-2cbb6b0eabd9.png)

## Autocorrelation + Seasonality + Trend

combined plot for Autocorrelation, seasonality and trend together

```python
amplitude = 10
slope = 2
series = autocorrelation_1(time, amplitude, seed = RANDOM_SEED) + seasonality(time, period = 50, amplitude = 150) + trend(time, slope)
plt.figure(figsize = (20, 6))
plot_series(time[:300], series[:300], color = "red")
plt.title("Autocorrelation + Seasonality + Trend Plot", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide10](https://user-images.githubusercontent.com/130429032/235585110-33a0ad91-24d8-4763-8d76-aa0c5bfb229f.png)

## Autocorrelation + Seasonality + Noise + Trend

Combined plot for Autocorrelation, seasonality, noise and trend together

```python
amplitude = 10
slope = 2
series = autocorrelation_1(time, amplitude, seed = RANDOM_SEED) + seasonality(time, period = 50, amplitude = 150) + trend(time, slope)
series += white_noise(time, noise_level = 100)
plt.figure(figsize = (20, 6))
plot_series(time[:300], series[:300], color = "red")
plt.title("Autocorrelation + Seasonality + Noise + Trend Plot", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide11](https://user-images.githubusercontent.com/130429032/235585114-e896803f-8700-4ce2-83b8-02e61a7d16cb.png)

# Break Point

Break Point Plot at time = 200

Combined plot for Autocorrelation, seasonality, noise, trend and break point together

```python
amplitude1 = 10
amplitude2 = 5
slope1 = 2
slope2 = -2
series1 = autocorrelation_2(time, amplitude1, seed = RANDOM_SEED) + seasonality(time, period = 50, amplitude = 150) + trend(time, slope1)
series2 = autocorrelation_2(time, amplitude2, seed = RANDOM_SEED) + seasonality(time, period = 50, amplitude = 2) + trend(time, slope2) + 750 + white_noise(time, 30)
series1[200:] = series2[200:]
series1 += white_noise(time, noise_level = 100, seed = RANDOM_SEED)
plt.figure(figsize = (20, 6))
plot_series(time[:300], series1[:300], color = "indigo")
plt.axvline(x = 200, color = "black")
plt.title("Autocorrelation + Seasonality + Noise + Trend + Break Point Plot", fontdict = {'fontsize' : 20})
plt.show()
```

![Beginners-Guide12](https://user-images.githubusercontent.com/130429032/235585117-4d6f3a9c-c44c-4b65-ba01-24f78b023aff.png)

- 이 코드는 여러 가지 시계열 패턴을 조합한 시계열 데이터를 생성하고 이를 시각화하는 코드입니다.
- 첫 번째 시계열 series1은 autocorrelation_2 함수를 이용하여 생성됩니다. 이 함수는 이전 시간 단계의 값을 현재 값에 더하는 자기회귀(AR)모형을 사용합니다. 이를 통해 시계열 데이터 내에서 관측치 간의 자기 상관관계를 가지게 됩니다.
- 두 번째 시계열 series2는 동일한 방법으로 생성됩니다. 다만, amplitude와 slope의 값이 다르며, 이후에 series1과 연결됩니다.
- 두 시계열에는 seasonality와 noise 가 추가됩니다. seasonality는 주기적인 패턴을 가진 반복 요소입니다. 이를 위해 period와 amplitude를 조절할 수 있습니다. noise는 white_noise 함수를 이용하여 생성합니다. 이는 임의로 생성된 값들로 시계열 데이터에 무작위로 더해집니다. 이를 통해 데이터 내에서 불규칙한 변동성을 가지게 됩니다.
- 마지막으로, 두 시계열을 결합합니다. 200번째 시간 단계를 기준으로 series1의 데이터가 series2로 교체됩니다. 이를 통해 breakpoint가 추가됩니다.