# 여러가지 유사도 측정 기법

---

## 1. 자카드 유사도

- 교집합 / 합집합의 비율을 바탕으로 대상의 유사도를 계산



### 1) 수식적 접근

```python
# 데이터 지정 및 띄워쓰기 토큰화
data1 = '안녕 나는 오늘 힘들어'.split()
data2 = '안녕 못해 나는 지금 너무 힘들어'.split()

# data1과 data2의 합집합
un = set(data1) | set(data2)

# data1과 data2의 교집합
intd = set(data1) & set(data2)

# 자카드 유사도 계산
len(intd) / len(un)
```

```
0.42857142857142855
```



### 2) 패키지 사용

1. import 및 예시코드

   ```python
   # 자카드 유사도 계산
   # 제약조건 : 반드시 대상은 0~1 사이의 값을 가져야한다.
   from sklearn.metrics import jaccard_score
   import numpy as np
   
   jaccard_score(np.array([0, 1, 0, 0]), np.array([0, 1, 1, 2]), average=None)
   ```

   ```
   array([0.33333333, 0.5       , 0.        ])
   ```

---

## 2. 유클리디안 유사도



### 1) 수식적 접근

1. 함수 생성

   ```python
   # 유클리드 거리 계산 함수
   def f(A, B):
       return np.sqrt(np.sum((A-B)**2))
   ```

2. 테스트 데이터

   ```python
   # 테스트할 데이터(A와 B를 각각 문장이라고 생각)
   A = np.array([0, 1, 2, 3, 4])
   B = np.array([1, 0, 1, 2, 3])
   ```

3. 거리 계산

   ```python
   # A와 B의 거리 연산
   f(A, B)
   ```

   ```
   2.23606797749979
   ```



### 2) 패키지 이용

1. 테스트 데이터

   ```python
   data1 = '안녕 나는 오늘 힘들어'
   data2 = '안녕 못해 나는 지금 너무 힘들어'
   ```

2. 토큰화 및 벡터화

   ```python
   # 토큰화 및 벡터화
   from sklearn.feature_extraction.text import TfidfVectorizer # 문장끼리의 관련성을 고려한 벡터화 패키지
   
   t_v = TfidfVectorizer()
   m_data = t_v.fit_transform([data1, data2])
   ```

3. 유클리드 거리 계산

   ```python
   from sklearn.metrics import euclidean_distances
   
   euclidean_distances(m_data[0:1], m_data[1:2])
   ```

   ```
   array([[1.0486415]])
   ```

4. 정규화 후 거리 계산

   ```python
   # 정규화 : 전체의 합으로 나누어서 비율로 나타냄(유사도 판명 목적) -> 계산의 용이성을 위함
   def e_f(m):
       return m / np.sum(m_data)
   
   e_data = e_f(m_data)
   euclidean_distances(e_data[0:1], e_data[1:2])
   ```

   ```
   array([[0.23884449]])
   ```

---

## 3. 멘허튼 유사도

- 유클리드 거리는 단어의 중요도에 관계없이 단순 거리를 측정하여 부정확한 결과를 야기할 수 있다.
- 맨허튼 유사도는 각 피쳐마다의 특징을 고려하기 때문에 조금 더 정확한 결과를 만들 수 있다.



### 1) 수식적 접근

1. 테스트 

   ```python
   a = np.array([-1, 2, 3])
   b = np.array([1, 3, -4])
   
   def f(A, B):
       return np.sum(abs(A-B))
   f(A, B)
   ```



### 2) 패키지 이용

1. 테스트 데이터

   ```python
   data1 = '안녕 나는 오늘 힘들어'
   data2 = '안녕 못해 나는 지금 너무 힘들어'
   ```

2. 토큰화 및 벡터화

   ```python
   # 토큰화 및 벡터화
   from sklearn.feature_extraction.text import TfidfVectorizer # 문장끼리의 관련성을 고려한 벡터화 패키지
   
   t_v = TfidfVectorizer()
   m_data = t_v.fit_transform([data1, data2])
   ```

3. 맨허튼 거리 계산

   ```python
   from sklearn.metrics.pairwise import manhattan_distances
   
   # 정규화 x일 때, 맨하탄 거리
   manhattan_distances(m_data[0:1], m_data[1:2])
   ```

   ```
   array([[2.38220441]])
   ```

4. 정규화 후 계산

   ```python
   def e_f(m):
       return m / np.sum(m_data)
   
   e_data = e_f(m_data)
   
   # 정규화된 데이터를 이용하여 유사도 계산
   manhattan_distances(e_data[0:1], e_data[1:2])
   ```

   ```
   array([[0.54258429]]))
   ```



**정규화를 해야하는 이유**

- 두 데이터 간의 차이가 더 명확해진다.
- 두 데이터 간의 유사성 판별이 더 명확해진다.