# 탐색적 데이터 분석 : 기술통계



## 1. 데이터 분석



### 1) pandas 내부 메소드

#### (1) describe()

- 수치형 데이터에 해당하는 열에 대해서 요약 데이터를 계산하여 나타낸다.
  - count : 데이터 개수
  - mean : 평균
  - std : 표준편차
  - min : 최소값
  - 25% : 1사분위수
  - 50% : 중앙값
  - 75% : 3사분위수
  - max : 최대값

- 데이터의 적정성을 확인하기 위함, 즉, 데이터의 특성을 파악하기 위함이다.
- ex. ``bike_data.describe()``

#### (2) colums

- 데이터프레임의 열 이름을 모두 출력
- ``bike_data.columns``

#### (3) head()

- 데이터의 첫 다섯 줄을 보여준다.
- ``bike_data.head()``
- ``bike_data.head(10)`` : 파라미터 전달을 통해서 몇 개의 줄을 볼 것인지 설정

#### (4) tail()

- 데이터의 마지막 다섯 줄을 보여준다
- ``bike_data.tail()``

#### (5) shape

- 데이터의 행과 열의 구조를 보여준다.
- ``bike_data.shape``

- 행/열의 순으로 보여준다.

#### (6) info()

- 데이터의 전반적인 정보를 보여준다.
- ``bike_data.info``
- null의 개수를 파악하여 결측치를 처리하는데 참고해야한다.



---

#### (7) 특정 통계 값을 계산하는 메소드

|  메소드  |             설명              |
| :------: | :---------------------------: |
|  count   | NA 값을 제외한 값의 수를 반환 |
| describe |    각 열에 대한 요약 통계     |
|   min    |            최소값             |
|   max    |            최대값             |
|   sum    |             합계              |
|   mean   |             평균              |
|   var    |             분삭              |
|   std    |           표준편차            |
|   skew   |       왜도(치우침 정도)       |
|   kurt   |       첨도(뾰족함 정도)       |
|  cumsum  |            누적 합            |



### 2) DataFrame의 특정 컬럼만 추출하여 보는 방법



#### (1) 점을 사용하는 방법

- ``bike_data.Distance``

- ``bike_data.Distance.sum()`` : 지정한 열의 합계



#### (2) 대괄호를 사용하는 방법

- ``bike_data['Distance']``
- ``bike_data['Distance'].sum()`` : 지정한 열의 합계



#### (3) 지정한 열의 값을 확인

- ``bike_data['Membership_type'].unique()``
- 해당 열에 있는 값을 중복을 제외하여 알려준다.



#### (4) 지정한 열의 값 별 개수 확인

- ``bike_data['Membership_type'].value_counts()``
- 각 값(항목) 별로 몇 개의 데이터가 있는지 그 개수를 알려준다.



#### (5) 지정한 열의 값 별 비율 확인

- ``bike_data['Membership_type'].value_counts(normalize=True)``
- 값 별 비율을 확인한다.



#### (6) 특정 열의 Null 값 확인

- ``bike_data.Momentum.isnull().sum()``
- 특정 열의 null 값의 개수를 확인한다.



#### (7) 특정 열의 특정 값을 확인

- ``bike_data[bike_data.Momentum == '\\N']``
- 대괄호 내에 조건을 입력한다.