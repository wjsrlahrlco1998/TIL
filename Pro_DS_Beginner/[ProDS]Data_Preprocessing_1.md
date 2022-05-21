# 데이터 전처리 : 결측치 이상치



## 1) 이상치

- 정의
  - 중심 경향성에서 멀리 떨어진 값

- 이상치 처리 방법
  - 절대적인 기준 x
  - 대표적으로 Carling, Tukey 방법
  - 분포 기반으로 처리도 가능



## 2) 결측치

- 정의
  - 값이 기록되지 않고 비어 있음
- 결측치 처리 방법
  - 절대적인 기준 x
  - 단순 제거, 특정 값으로 대체
  - 분석 데이터에서 결측치가 차지하는 비중이 낮으면 단순 제거하는 경우가 많다



## 3) 결측치 확인 함수

- pandas

  - ``isna(), isnull()`` : 결측치가 존재하면 True 반환

  - ``notna(), notnull()`` : 결측치가 존재하면 False 반환

  - fillna() : 결측치 채우기

    - ``df_na.fillna(value = {'Sepal_Length' : 999,   "Sepal_Width" : 999})`` : 열을 선택하여 지정 값으로 채울 수 있다.
    - df_na.fillna(value = 999) : 열을 지정하지 않고 채움

    

## 4) 결측치 제거 함수

- pandas
  - ``dropna()`` : 디폴트 값은 how='any'로 결측치가 존재하는 행을 삭제, how='all'로 설정하면 모든 행이 결측치인 경우 삭제



## 5) 분위수 확인

```python
print(df["Sepal_Length"].quantile(q=0.25)) # 제 1 사분위수
print(df["Sepal_Length"].quantile(q=0.50)) # 제 2 사분위수
print(df["Sepal_Length"].quantile(q=0.75)) # 제 3 사분위수
```

