---
layout: single
title:  "Creating a Streamlit Web Page Second Practice"
categories: Python
tag: [Python, Streamlit]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---
<br/>
# Streamlit Library 연습하기
---
```python
fig = px.scatter(data_frame = iris, x = 'sepal_length', y = 'sepal_width')
st.plotly_chart(fig)
```
Plotly Express 라이브러리를 사용하여 iris 데이터셋의 sepal_length와 sepal_width 사이의 관계를 나타내는 산점도를 만드는 코드입니다. x축은 sepal_length이고 y축은 sepal_width입니다.   
- st.plotly_chart(fig)
Streamlit에서 Plotly 그래프를 표시하는 데 사용하는 함수입니다. 이 함수는 Plotly의 plot()함수와 매우 유사한 인수를 사용합니다.   
<br/>

```python
choice = st.selectbox('품종', iris['species'].unique())
result = iris[iris['species'] == choice].reset_index(drop = True)
col1, col2 = st.columns([0.5, 0.5], gap = 'large')
with col1:
    fig2, ax = plt.subplots()
    sns.scatterplot(result, x = 'petal_length', y = 'sepal_width')
    st.pyplot(fig2)
with col2:
    fig3, ax = plt.subplots()
    ax.scatter(x = result['sepal_length'], y = result['sepal_width'])
    st.pyplot(fig3)
```
<br/>
- **st.selectbox()**   
Streamlit에서 사용하는 함수 중 하나입니다. 이 함수는 레이블과 옵션을 인수로 받아 선택 상자를 생성합니다. 그리고 사용자가 옵션 중 하나를 선택할 수 있또록 합니다.   
- **unique()**
pandas library에서 제공하는 함수 중 하나입니다. pandas Series나 DataFrame에서 고유한 값을 반환합니다. 다시 말해 중복된 행을 제거합니다.   
- **reset_index()**   
pandas DataFrame에서 인덱스를 재설정하는 함수입니다. 기존 인덱스를 제거하고 새로운 정수 인덱스를 할당합니다. drop 매개변수는 기존 인덱스를 DataFrame의 열로 유지할지 여부를 결정합니다. drop = True로 설정하면 기존 인덱스가 DataFrame의 열로 유지되지 않습니다.   
- **st.columns()**   
Streamlit 앱에서 열을 만드는 데 사용됩니다. 인수로 전달된 숫자 리스트에 따라 열을 생성합니다. 각 숫자는 열의 너비를 나타내며 숫자가 클수록 열이 더 넓어집니다. gap = 'large'는 열 사이의 간격을 조정하는데 사용됩니다. 'small', 'medium', 'large'와 같은 문자열이 전달될 수 잇으며 이 문자열에 따라 열 사이의 간격이 조정됩니다.   
- **plt.subplots()**   
matplotlib 라이브러리에서 사용되는 함수입니다. 이 함수는 여러 개의 서브플롯을 생성하는 데 사용됩니다. 두 개의 값을 반환합니다. 첫 번째 값은 figure 객체이며, 두 번째 값은 axes 객체입니다.   
- **sns.scatterplot()**   
seaborn 라이브러리에서 산점도를 그리는 데 사용되는 함수입니다. 다양한 매개변수를 지원하며 hue 매개변수를 사용하여 데이터의 다른 특성에 따라 색상을 지정할 수 있습니다.   
- **st.pyplot()**   
streamlit 라이브러리에서 생성된 그래프를 Streamlit 어플리케이션에 표시하는 데 사용됩니다.   
- **ax.scatter()**   
matplotlib 라이브러리에서 산점도를 그리는 데 사용됩니다. x와 y 배열을 사용하여 산점도를 생성하며 다양한 매개변수를 지원합니다. 예를 들어, s 매개변수를 사용하여 점의 크기를 지정할 수 있습니다.   
<br/>

```python
def run_eda_app():
    st.subheader("탐색적 자료 분석")

    iris = pd.read_csv('data/iris.csv')
    st.markdown('## IRIS 데이터 확인')
    st.write(iris) # print()

    # 메뉴 지정
    submenu = st.sidebar.selectbox('하위 메뉴', ['기술통계량', '그래프분석', '통계분석'])
    if submenu == '기술통계량':
        st.dataframe(iris)

        with st.expander('데이터 타입'):
            result1 = pd.DataFrame(iris.dtypes)
            st.write(result1)
        with st.expander('기술 통계량'):
            result2 = iris.describe()
            st.write(result2)

        with st.expander("타깃 빈도 수 확인"):
            st.write(iris['species'].value_counts())

    elif submenu == '그래프분석':
        st.title("Title")
        with st.expander('산점도'):
            fig1 = px.scatter(iris, x = 'sepal_width', y = 'sepal_length', color = 'species', size = 'petal_width', hover_data = ['petal_length'])
        st.plotly_chart(fig1)

    # layouts
    col1, col2 = st.columns(2)
    with col1:
        st.title('Seaborn')
        fig2, ax = plt.subplots()
        sns.scatterplot(data = iris, x = 'sepal_width', y = 'sepal_length')
        st.pyplot(fig2)

    with col2:
        st.title('matplotlib')
        fig3, ax = plt.subplots()
        ax.scatter(x = 'sepal_width', y = 'sepal_length', data = iris)
        st.pyplot(fig3)

    # Tabs
    tab1, tab2 = st.tabs(['탭1', '탭2'])
    with tab1:
        st.write('탭1')
        choice = st.selectbox('종', iris['species'].unique)
        result = iris[iris['species'] == choice].reset_index(drop = True)
        fig4, ax = plt.subplots()
        fig4 = px.scatter(result, x = result['sepal_length'], y = result['sepla_width'])
        st.plotly_chart(fig4, use_container_width = True)

    with tab2:st.write('탭2'):
        consult = pd.read_csv('data/consulting.csv', encoding = 'cp949')
        st.line_chart(consult)

    elif submenu == '통계분석':
        pass
    else:
        st.warning("뭔가 없어요!")
```
<br/>
- **st.sidebar.selectbox()**   
이 함수는 사이드바에 드롭다운 메뉴를 생성하는데 사용됩니다. 또한 다양한 매개변수를 지원합니다. 예를 들어 key 매개변수를 사용하여 위젯의 고유 키를 지정할 수 있습니다.   
- **st.expander()**   
이 함수는 확장 가능한 섹션을 생성하는 데 사용됩니다. 또한 다양한 매개변수를 지정합니다. 예를 들어 expanded 매개변수를 사용하여 섹션을 기본적으로 확장할지 축소할지 지정할 수 있습니다.   
- **px.scatter()**   
Plotly 라이브러리에서 산점도를 그리는데 사용합니다. 또한 다양한 매개변수를 지정하며 예로, color 매개변수를 사용하여 점의 색상을 지정할 수 있습니다.   
- **st.plotly_chart()**
Streamlit 라이브러리에서 Plotly 그래프를 Streamlit 앱에 표시하는 데 사용됩니다. 또한 다양한 매개변수를 지원하며 예로, use_container_width 매개변수를 사용하여 그래프의 너비를 컨테이너의 너비에 맞게 조정할 수 있습니다.   
- **st.columns()**   
Streamlit 라이브러리에서 열을 생성하는 데 사용합니다. 다양한 매개변수를 지원하며 예로, width 매개변수를 사용하여 열의 너비를 지정할 수 있습니다.   
- **st.line_chart()**   
Streamlit 라이브러리에서 선 그래프를 그리는 데 사용됩니다. 다양한 매개변수를 지원하며 예로, use_container_width 매개변수를 사용하여 그래프의 너비를 컨테이너의 너비에 맞게 조정할 수 있습니다.   
<br/>

```python
data = pd.read_csv('data/iris.csv')
le - LabelEncoder()
print(le.fit(data['species']))
data['species'] = le.fit_transform(data['species'])
print(le.classes_)

x = data.drop(columns = ['species'])
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

model = LogisticRegression()
model.fit(X_train, y_train)

# 모델 만들고 배포 (Export)
model_file = open("models/logistic_regression_model_iris_230425.pkl", "wb")
joblib.dump(model, model_file)
model_file.close()
```
<br/>
- **LabelEncoder()**   
scikit-learn 라비르러리에서 레이블을 숫자로 인코딩하는 데 사용됩니다. 이 함수는 다양한 매개변수를 지원하지 않습니다.   
- **le.fit()**   
머신러닝에서 사용되는 함수 중 하나며 입력 데이터와 출력 데이터를 받아서 머신러닝 모델을 학습시키기 위해 사용됩니다. 또한 입력 데이터와 출력 데이터를 숫자형으로 변환합니다.   
- **le.classes_**   
LabelEncoder 클래스의 속성 중 하나며 LabelEncoder 클래스가 변환한 범주형 데이터의 클레스 레이블을 반환합니다.   
- **train_test_split()**   
머신러닝에서 사용하는 함수 중 하나로써 데이터를 학습용 데이터와 테스트용 데이터로 나누어주는 역할을 합니다. 입력 데이터와 출력 데이터를 무작위로 섞은 후에 학습용 데이터와 테스트용 데이터로 나누어줍니다. 기본적으로 학습용 데이터와 테스트용 데이터를 75%와 25%로 나누어줍니다.   
- **open()**   
파이썬 내장 함수 중 하나로 파일을 열 때 사용됩니다. 읽기 모드(r), 쓰기 모드(w), 추가 모드(a) 등이 있습니다. w를 입력하면 파일을 쓰기 모드로 열고 b를 입력하면 바이너리 모드로 열 수 있습니다.   
- **joblib.dump()**   
파이썬에서 객체를 저장할 때 사용하는 함수 중 하나입니다. 입력값으로 모델과 파일 이름을 받습니다. 이 함수는 모델을 파일로 저장합니다. 파일 이름을 입력받아서 해당 파일에 모델을 저장합니다.   
<br/>

```python
def run_ml_app():
    st.title("머신러닝 페이지")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("수치를 입력해주세요!")
        sepal_length = st.select_Slider("Sepal Length", options = np.arange(1,11))
        sepal_width = st.select_Slider("Sepal Width", options = np.arange(1,11))
        petal_length = st.select_Slider("Petal Length", options = np.arange(1,11))
        petal_width = st.select_Slider("Petal Width", options = np.arange(1,11))

        sample = [sepal_length, sepal_Width, petal_length, petal_width]
        st.write(sample)

    with col2:
        st.subheader("모델 결과를 확인해주세요!")
        new_df = np.array(sample).reshape(1, -1)
        st.write(new_df.shape, new_df.ndim)
        MODEL_PATH = 'models/logistic_regression_model_iris_230425.pkl'
        model = joblib.load(open(os.path.join(MODEL_PATH), 'rb'))
        prediction = model.predict(new_df)
        pred_prob = model.predict_proba(new_df)
        st.write(predction)
        st.write(pred_prob)
        if predction == 0:
            st.success("Setosa 종입니다.")
            pred_proba_scores = {"Setosa 확률": pred_prob[0][0] * 100,
            "Versicolor 확률": pred_prob[0][1] * 100,
            "Virginica 확률": pred_prob[0][2] * 100}
            st.write(pred_proba_scores)
            st.image('...')
        elif prediction == 1:
            st.success("Versicolor 종입니다.")
            pred_proba_scores = {"Setosa 확률": pred_prob[0][0] * 100,
            "Versicolor 확률": pred_prob[0][1] * 100,
            "Virginica 확률": pred_prob[0][2] * 100}
            st.write(pred_proba_scores)
            st.image('...')

        elif prediction == 2:
            st.success("Virginica 종입니다.")
            pred_proba_scores = {"Setosa 확률": pred_prob[0][0] * 100,
            "Versicolor 확률": pred_prob[0][1] * 100,
            "Virginica 확률": pred_prob[0][2] * 100}
            st.write(pred_proba_scores)
            st.image('...')
        else:
            st.warning("판별 불가")
```
<br/>
- **st.select_Slider()**   
이 함수는 슬라이더를 생성하는 함수로써 입력값으로 슬라이더의 최소값, 최대값, 기본값, 스텝 등을 받습니다.   
- **np.arange()**   
이 함수는 일정한 간격으로 배열을 생성하는 역할을 합니다. 입력값으로 시작값, 종료값, 간격 등을 받습니다.   
- **reshape()**   
이 함수는 NumPy에서 제공하는 배열의 차원을 변경하는 함수입니다. 입력값으로 변경하고자 하는 배열의 모양을 받습니다.   
- **.ndim**
이 함수는 배열의 차원 수를 반환하는 함수입니다.   
- **joblib.load()**   
이 함수는 저장된 객체를 로드하는 역할을 하며 입력값으로 로드하고자 하는 파일의 경로를 받습니다.   
- **predict()**   
머신러닝에서 제공하는 함수 중 하나며 모델을 사용하여 예측을 수행하는 역할을 합니다. 입력값으로 예측하고자 하는 데이터를 받고 예측 결과를 반환합니다.   
