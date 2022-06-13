# 비계층적 군집분석(K-means Clustering)

## 1. k-means

- 임의의 k개의 점을 기반으로 가까운 거리의 데이터를 묶는 것과 더불어 평균을 활용하는 군집분석 기법
- 군집 개수(k)를 확정하기 위해 여러 번의 시행착오 필요
- 결과 고정을 위해 seed 설정 필요

## 2. 주요 함수 및 메서드

- sklearn - MinMaxScaler()
  - MinMax 정규화를 실시하는 sklearn의 함수
  - fit() 메서드로 규칙 모델을 만들고 transform() 함수로 변환을 실시
- sklearn - StandardScaler()
  - 표준화를 실시하는 sklearn의 함수
  - fit() 메서드로 규칙 모델을 만들고 transform() 함수로 변환을 실시
- sklearn - KMeans()
  - k-means 군집분석을 실시하는 sklearn 함수
  - n_clusters, max_iter, random_state에 각각 군집 개수, 최대 반복 연산, 결과 고정 설정 가능
  - KMeans() 함수의 fit() 메서드에 데이터를 할당하여 학습 진행
  - 결과 객체의 cluster_centers_와 labels_ 어트리뷰트로 군집 중심과 각 행의 군집 번호 확인 가능

## 3. 코드

### * 패키지

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
```

### 1) MinMaxScaler() - StandardScaler()와 사용법 같음

```python
# 데이터 로드
df = pd.read_csv("iris.csv")
df.head(2)
```

![image-20220613230940861]([ProDS]Kmeans_Clustering.assets/image-20220613230940861.png)

```python
# 데이터 분리 -> 기준에 따른 차이 확인을 위함
df_1 = df.head()
df_2 = df.tail(1)
```

![image-20220613231022131]([ProDS]Kmeans_Clustering.assets/image-20220613231022131.png)

![image-20220613231026358]([ProDS]Kmeans_Clustering.assets/image-20220613231026358.png)

```python
nor_minmax = MinMaxScaler().fit(df_1.iloc[:, :-1])
nor_minmax.transform(df_2.iloc[:, :-1])
```

![image-20220613231103186]([ProDS]Kmeans_Clustering.assets/image-20220613231103186.png)

```python
# 학습의 기준에 주의해야한다.
nor_minmax = MinMaxScaler().fit(df_2.iloc[:, :-1])
nor_minmax.transform(df_2.iloc[:, :-1])
```

![image-20220613231114488]([ProDS]Kmeans_Clustering.assets/image-20220613231114488.png)

```python
# 데이터 프레임 변환
df_minmax = pd.DataFrame(nor_minmax.transform(df_1.iloc[:, :-1]), columns=df_1.columns[:4])
df_minmax
```

![image-20220613231130012]([ProDS]Kmeans_Clustering.assets/image-20220613231130012.png)

### 2) KMeans()

```python
model = KMeans(n_clusters=3, random_state=123).fit(df.iloc[:, :-1])
model
```

![image-20220613231148558]([ProDS]Kmeans_Clustering.assets/image-20220613231148558.png)

```python
# 분류 확인
model.labels_
```

![image-20220613231204561]([ProDS]Kmeans_Clustering.assets/image-20220613231204561.png)

```python
# 중심 값 확인
model.cluster_centers_
```

![image-20220613231219839]([ProDS]Kmeans_Clustering.assets/image-20220613231219839.png)

```python
# 클러스터 별 평균 값 확인
df["cluster"] = model.labels_
df.groupby("cluster").mean()
```

![image-20220613231239350]([ProDS]Kmeans_Clustering.assets/image-20220613231239350.png)

## 4. 문제

### Q1. BMI가 0이 아닌 사람 데이터를 대상으로 k-means 군집 분석을 실시하는 경우 군집 개수가 가장 큰 군집의 Insulin 평균은 얼마인가?

```python
df = pd.read_csv("diabetes.csv")
df.head(2)
```

![image-20220613231257887]([ProDS]Kmeans_Clustering.assets/image-20220613231257887.png)

```python
df = df.loc[df.BMI != 0]
df.head(2)
```

![image-20220613231306863]([ProDS]Kmeans_Clustering.assets/image-20220613231306863.png)

```python
model = KMeans(n_clusters=4, random_state=123).fit(df)
df["cluster"] = model.labels_
df.groupby("cluster")["Insulin"].mean()
```

![image-20220613231321771]([ProDS]Kmeans_Clustering.assets/image-20220613231321771.png)

```python
df.cluster.value_counts()
```

![image-20220613231330690]([ProDS]Kmeans_Clustering.assets/image-20220613231330690.png)

답 : 4.103194

### Q2. BMI가 0이 아닌 사람 데이터를 대상으로 k-means 군집 분석을 실시하는 경우 군집 개수가 가장 큰 군집의 나이 평균은 얼마인가?

```python
df = pd.read_csv("diabetes.csv")
df.head(2)
```

![image-20220613231350192]([ProDS]Kmeans_Clustering.assets/image-20220613231350192.png)

```python
df_sub = df.loc[df.BMI != 0,]
df_sub.head(2)
```

![image-20220613231359033]([ProDS]Kmeans_Clustering.assets/image-20220613231359033.png)

```python
df_sub = MinMaxScaler().fit_transform(df_sub)
df_sub
```

![image-20220613231408858]([ProDS]Kmeans_Clustering.assets/image-20220613231408858.png)

```python
df_sub = pd.DataFrame(df_sub, columns=df.columns)
df_sub.head(2)
```

![image-20220613231419108]([ProDS]Kmeans_Clustering.assets/image-20220613231419108.png)

```python
model = KMeans(n_clusters=4, random_state=123).fit(df_sub)
df_sub2 = df.loc[df.BMI != 0,]
df_sub2.head(2)
```

![image-20220613231437085]([ProDS]Kmeans_Clustering.assets/image-20220613231437085.png)

```python
df_sub2["cluster"] = model.labels_
df_sub2.head(2)
```

![image-20220613231445835]([ProDS]Kmeans_Clustering.assets/image-20220613231445835.png)

```python
df_sub2.groupby("cluster")["Age"].mean()
```

![image-20220613231454724]([ProDS]Kmeans_Clustering.assets/image-20220613231454724.png)

```python
df_sub2.cluster.value_counts()
```

![image-20220613231504930]([ProDS]Kmeans_Clustering.assets/image-20220613231504930.png)

### Q3. BMI가 0이 아닌 사람 데이터를 대상으로 k-means 군집 분석을 실시하고 군집의 중심점간 유클리드 거리가 가장 가까운 그룹간 거리는?

```python
df = pd.read_csv("diabetes.csv")
df.head(2)
```

![image-20220613231517005]([ProDS]Kmeans_Clustering.assets/image-20220613231517005.png)

```python
df_sub = df.loc[df.BMI != 0,]
df_sub.head(2)
```

![image-20220613231529585]([ProDS]Kmeans_Clustering.assets/image-20220613231529585.png)

```python
model = KMeans(n_clusters=3, random_state=123).fit(df_sub)
df_centers = pd.DataFrame(model.cluster_centers_, columns=df_sub.columns)
df_centers
```

![image-20220613231540231]([ProDS]Kmeans_Clustering.assets/image-20220613231540231.png)

```python
df_centers = df_centers.transpose()
df_centers
```

![image-20220613231551436]([ProDS]Kmeans_Clustering.assets/image-20220613231551436.png)

```python
print(sum((df_centers.iloc[:, 0] - df_centers.iloc[:, 1]) **2) **0.5)
print(sum((df_centers.iloc[:, 1] - df_centers.iloc[:, 2]) **2) **0.5)
print(sum((df_centers.iloc[:, 0] - df_centers.iloc[:, 2]) **2) **0.5)
```

![image-20220613231601025]([ProDS]Kmeans_Clustering.assets/image-20220613231601025.png)

답 : 146