---
layout: single
title:  "To practice creating a Streamlit web page"
categories: Python
tag: [Python, Streamlit]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

# Streamlit Library 연습하기

```python
import streamlit as st
import pandas as pd
from PTL import Image
```
- **import streamlit as st**   
streamlit은 데이터 분석, 시각화 및 상호작용을 위한 웹 어플리케이션을 빠르게 만드는 파이썬 라이브러리 입니다.   
<br/>  
- **import pandas as pd**   
pandas는 데이터 분석을 위한 파이썬 라이브러리입니다.   
<br/>
- **from PTL import Image**   
PTL은 Python Template Library의 약자로, Python에서 사용되는 템플릿 라이브러리입니다.   
PTL Image는 PTL라이브러리의 Image 모듈입니다. 이미지 처리를 위한 다양한 기능을 제공합니다.   
이미지를 불러오고 저장할 수 있으며 이미지를 회전하거나 크기를 조절하는 등의 작업을 할 수 있습니다.   
<br/>
```python
st.title("Hello World") # title
st.text("this is so {}".format("good")) # text
st.header("This is Header") # Header
st.subheader("This is subHeader") # Sub Header
st.markdown("This is Markdown") # Markdown
```
<br/>
- **st.title()**   
텍스트를 제목 형식으로 표시합니다.
<br/>
- **st.text()**   
고정폭 글꼴로 표시되는 서식이 지정된 텍스트를 작성합니다.   
고정폭 글꼴은 각 글자가 동일한 너비를 차지하는 글꼴입니다. 한글 글꼴의 경우 한글 글자의 폭은 사실 영문 글자의 폭의 두 배를 차지합니다.
<br/>
- **st.header()**   
헤더를 생성하는 함수입니다. 이 함수는 문자열을 인자로 반당서 해당 문자열을 헤더로 출력합니다.
<br/>
- **st.subheader()**   
서브헤더를 생성하는 함수입니다. 이 함수는 문자열을 인자로 받아서 해당 문자열을 서브헤더로 출력합니다.
<br/>
- **st.markdown()**   
마크다운을 생성하는 함수입니다. 이 함수는 문자열을 인자로 받아서 해당 문자열을 마크다운으로 출력합니다.   
<br/>
```python
st.success('성공')
st.warning('경고')
st.info('정보와 관련된 탭')
st.error('에러 메세지')
st.exception('예외처리')
```
<br/>
- **st.success()**   
성공 메세지를 생성하는 함수입니다.
<br/>
- **st.warning()**   
경고 메세지를 생성하는 함수입니다.
<br/>
- **st.info()**   
정보 메세지를 생성하는 함수입니다.
<br/>
- **st.error()**   
오류 메세지를 생성하는 함수입니다.
<br/>
- **st.exception()**   
예외 메세지를 생성하는 함수입니다. 예외 객체의 정보를 예외 메세지로 출력합니다.   
<br/>
```python
st.write('일반 텍스트')
st.write(1+2)
st.write(dir(str))
st.title(':sunglasses') # 이모티콘
# Help
st.help(range)
st.help(st.title)
```
<br/>
- **st.write()**   
다양한 데이터 타입을 출력하는 함수입니다.   
이 함수는 문자열, 숫자, 데이터 프레임, 스타일이 적용된 데이터 프레임 등 다양한 데이터 타입을 인자로 받아서 출력합니다.   
- **st.help()**   
객체에 대한 도움말과 정보를 출력하는 함수입니다.   
이 함수는 객체의 이름, 타입, 값, 시그니처, 도움말, 멤버 변수 및 메소드의 값/ 도움말을 출력합니다.   
<br/>
```python
iris = pd.read_csv('data/iris.csv') # iris 파일 불러오기
st.title("IRIS TABLE")
st.dataframe(iris, 500, 100) # Width, Height
st.title('table()')
st.table(iris)
st.title('write()')
st.write(iris)
myCode = """
def hello()
    print("hi")
"""
st.code(myCode, language = "Python")
```
<br/>
- **st.dataframe()**   
데이터프레임을 인자로 받아서 해당 데이터프레임을 인터렉티브한 테이블로 출력합니다.   
- **st.table()**   
데이터프레임을 테이블로 출력하는 함수입니다.   
데이터프레임의 모든 열과 행을 보여줍니다.   
데이터프레임의 크기에 따라 자동으로 크기를 조정합니다.   
- **st.code()**   
코드를 출력하는 함수입니다. 코드를 보기 좋게 출력하고, 코드의 언어를 지정할 수 있습니다.   
<br/>
```python
name = 'Sukyung'
if st.button('Submit'):
    st.write(f'name: {name.upper()}')
# RadioButton
s_state = st.radio('Status', ('활성화', '비활성화'))
if s_state == '활성화':
    st.success('활성화 상태')
else:
    st.error('비활성화 상태')
# Check Box
if st.checkbox('show/hide'):
    st.text('무언가를 보여줘!')
# Select Box
p_lans = ['Python', 'Julia', 'Go', 'Rust', 'JAVA', 'R']
choice - st.selectbox('프로그래밍 언어', p_lans)
st.write(f'{choice} 언어를 선택함')
# multiple selection
lans = ("영어", "일본어", "중국어", "독일어")
myChoice = st.multiselect("언어 선택", lans, default = "중국어")
st.write('선택', myChoice)
# Slider
age = st.slider('나이', 1, 120)
st.write(age)
```
<br/>
- **st.button()**   
버튼을 생성하는 함수입니다. 버튼을 생성하고 버튼이 클릭되었는지 여부를 반환합니다.   
- **st.radio()**   
라디오 버튼을 생성하는 함수입니다. 이 함수는 라디오 버튼을 생성하고 사용자가 선택한 값을 반환합니다.   
- **st.checkbox()**   
이 위젯은 체크박스를 보여주며 체크박스의 상태를 반환합니다.    
    - label: 체크박스 옆에 표시되는 라벨입니다.
    - value: 체크박스의 초기값입니다.
    - help: 체크박스 아래에 표시되는 도움말입니다.
    - on_change: 체크박스의 상태가 변경될 때 호출되는 콜백 함수입니다.
    - args: 콜백 함수에 전달할 위치 인수입니다.
    - kwargs: 콜백 함수에 전달할 키워드 인수입니다.
    - disabled: 체크박스를 비활성화할지 여부입니다.
    - label_visibility: 라베을 언제 표시할지 여부입니다.   
- **st.selectbox()**   
사용자에게 선택지를 보여주며 선택한 값을 반환합니다.
    - label: select 위젯 옆에 표시되는 라벨입니다.
    - options: 사용자에게 보여줄 선택지입니다.
    - index: select 위젯의 초기값입니다.
    - format_func: 선택지의 표시 방법을 지정하는 함수입니다.
    - key: 위젯의 고유한 식별자입니다.
    - help: select 위젯 아래에 표시되는 도움말입니다.
    - on_change: select 위젯의 상태가 변경될 때 호출되는 콜백 함수입니다.
    - args: 콜백 함수에 전달할 위치 인수입니다.
    - kwargs: 콜백 함수에 전달할 키워드 인수입니다.
    - disabled: select 위젯을 비활성화할지 여부입니다.
    - label_visibility: 라벨을 언제 표시할지 여부입니다.   
- **st.multiselect()**   
사용자에게 여러 선택지를 보여주며 사용자가 선택한 값을 반환합니다.
    - default: multiselect 위젯의 초기값입니다.
    - disabled: multiselect 위젯을 비활성화할지 여부입니다.
    - max_selections: 최대 선택 가능한 항목 수 입니다.
    - label, options, format_func, help, on_change, args, kwargs는 selectbox()함수와 같습니다.   
- **st.slider()**   
사용자에게 값을 선택하도록 허용하며, 사용자가 선택한 값을 반환합니다.
    - label: slider 위젯 옆에 표시되는 라벨입니다.
    - min_value: 슬라이더의 최소값입니다.
    - max_value: 슬라이더의 최대값입니다.
    - value: 슬라이더의 초기값입니다.
    - step: 슬라이더의 간격입니다.
    - format: 슬라이더의 값 표시 방법을 지정하는 문자열 형식입니다.
    - key: 위젯의 고유한 식별자입니다.
    - help: slider 위젯 아래에 표시되는 도움말입니다.   
<br/>
```python
# 이미지 가져오기
img = Image.open('data/image_01.jpg')
st.image(img)
url = 'data/image_01.jpg'
st.image(url)
# 비디오 출력
with open('data/secret_of_success.mp4', 'rb') as rb:
    video_file = rb.read()
    st.video(video_file, start_time = 1)
# 오디오 출력
with open('data/song.mp3', 'rb') as rb:
    audio_file = rb.read()
    st.audio(audio_file, format = "audio/mp3")
```
<br/>
- **Image.open()**   
PIL에서 제공하는 함수로 이미지 파일을 열고 PIL 이미지 객체를 반환합니다.   
- **st.image()**   
이 이미지 위젯은 이미지를 표시하고 이미지를 클릭하면 새 창에서 이미지를 열 수 있습니다.   
    - image: 표시할 이미지입니다.
    - caption: 이미지 아래에 표시할 캡션입니다.
    - width: 이미지의 너비입니다.
    - use_column_width: True로 설정하면 열의 너비에 맞게 이미지가 조정됩니다.
    - clamp: True로 설정하면 이미지가 자르기 모드로 표시됩니다.   
- **st.video()**   
이 위젯은 비디오를 표시하고 비디오를 클릭하면 새 창에서 비디오를 열 수 있습니다.   
    - data: 표시할 비디오 데이터입니다.
    - format: 데이터의 형식입니다.
    - start_time: 비디오의 시작 시간입니다.
    - width: 비디오의 너비입니다.
    - height: 비디오의 높이입니다.   
- **st.audio()**
이 위젯은 오디오를 표시하고, 오디오를 클릭하면 새 창에서 오디오를 열 수 있습니다.   
    - data: 표시할 오디오 데이터입니다.
    - format: 데이터의 형식입니다.   
<br/>
<br/>

**streamlit 오늘 공부한 내용을 정리해봤습니다.**