---
layout: single
title:  "Streamlit 웹 페이지 생성 연습하기"
categories: Python
tag: [Python, Streamlit]
toc: true
author_profile: true
sidebar:
    nav: "sidebar-category"
---

# Streamlit code 연습하기

```python
import streamlit as st
import pandas as pd
from PTL import Image
```
- import streamlit as st   
streamlit은 데이터 분석, 시각화 및 상호작용을 위한 웹 어플리케이션을 빠르게 만드는 파이썬 라이브러리 입니다.   
- import pandas as pd   
pandas는 데이터 분석을 위한 파이썬 라이브러리입니다.   
- from PTL import Image   
PTL은 Python Template Library의 약자로, Python에서 사용되는 템플릿 라이브러리입니다.   
PTL Image는 PTL라이브러리의 Image 모듈입니다. 이미지 처리를 위한 다양한 기능을 제공합니다.   
이미지를 불러오고 저장할 수 있으며 이미지를 회전하거나 크기를 조절하는 등의 작업을 할 수 있습니다.