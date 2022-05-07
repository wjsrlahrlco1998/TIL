# 데이터 업로드



## 1. 패키지



### 1) os 패키지

- 운영체제의 기능을 사용하여 알 수 있는 정보를 파이썬 안에서 사용할 수 있도록 연결해주는 프로그램을 담고 있는 패키지
- ``import os``



### 2) pandas 패키지

- 엑셀의 표같이 구성된 데이터를 다루기 쉽게 만들어주는 강력한 패키지
- ``import pandas as pd``



## 2. 데이터 불러오기



### 1) csv 파일 읽기

- 파일을 읽기에 앞서 Encoding 방식을 확인해야한다.
- Encoding 방식은 해당 파일을 Text로 열러 우측 하단에 Encoding 방식을 확인한다.
  - 해결방법 - 1 : 메모장을 다른이름으로 저장할 때 인코딩 방식을 'utf-8'로 지정
  - 해결방법 - 2 : ``bike_data = pd.read_csv('bike_usage_0.csv', encoding='ANSI')`` 로 직접 인코딩 방식 지정



### 2) null 값 확인

- ``bike_data.isnull()`` 을 통해서 확인(각 셀마다 확인)
- ``bike_data.isnull().sum()`` 을 통해서 열별로 null 값이 들어있는 cell의 개수를 알려준다.



### 3) txt 파일 읽기

- sep 옵션으로 구분자를 지정하여 읽어온다.

- ``population = pd.read_csv('population_by_Gu.txt', sep='\t')``

  

