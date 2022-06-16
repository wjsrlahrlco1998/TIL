# KNN(K-Nearest Neighbor)

## 1. KNN 개요
- KNN 분류
    - 새로운 값은 기존의 데이터를 기준으로 가장 가까운 k개의 최근접 값을 기준으로 분류
    - k는 동률의 문제 때문에 짝수는 되도록이면 피하는 것이 좋음
    - k가 1에 가까울수록 과적합, k가 클수록 과소적합이 되기 때문에 적절한 k값 선정 필요
- KNN 회귀
    - 기본 개념은 분류모델과 같으며 k개의 인접한 자료의 (가중)평균으로 예측

## 2. 주요 함수 및 메서드
- sklearn - KNeighborsClassifier()
    - KNN 분류 모델을 학습하기 위한 sklearn의 함수
    - n_neighbors 인자에 학습 시 고려할 이웃 데이터의 개수를 지정
    - n_neighbors가 1에 가까울수록 과적합되며 커질수록 과소적합되는 경향 존재
    - KNeighborsClassifier() 함수의 fit() 메서드에 독립변수와 종속변수 할당

## 3. 코드


```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
```


```python
df = pd.read_csv("iris.csv")
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["is_setosa"] = (df.Species == "setosa") + 0
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
      <th>is_setosa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_c = KNeighborsClassifier(n_neighbors=3)
model_c.fit(X = df.iloc[:, :4], y = df.is_setosa)
model_c
```




    KNeighborsClassifier(n_neighbors=3)




```python
model_c.predict(df.iloc[:, :4])[:5]
```




    array([1, 1, 1, 1, 1])




```python
from sklearn.metrics import accuracy_score
```


```python
accuracy_score(y_true = df["is_setosa"], y_pred = model_c.predict(df.iloc[:, :4]))
```




    1.0




```python
# 회귀 KNN
model_r = KNeighborsRegressor(n_neighbors=3)
model_r.fit(X = df.iloc[:, :3], y = df["Petal.Width"])
model_r
```




    KNeighborsRegressor(n_neighbors=3)




```python
pred_r = model_r.predict(df.iloc[:, :3])
pred_r[:5]
```




    array([0.26666667, 0.2       , 0.23333333, 0.2       , 0.16666667])




```python
from sklearn.metrics import mean_squared_error as mse
```


```python
mse(y_true = df["Petal.Width"], y_pred = pred_r)
```




    0.018651851851851857




```python
mse(y_true = df["Petal.Width"], y_pred = pred_r) ** 0.5
```




    0.13657178278052848



## 4. 문제

### Q1. 당뇨 발생 여부를 예측하기 위해 임신 횟수, 혈당, 혈압을 사용할 경우 그 정확도는 얼마인가?
- diabates.csv 사용
- 데이터를 학습7, 평가3의 비율로 분할하시오. seed = 123
- 설정은 모두 기본값으로 하시오.


```python
df = pd.read_csv("diabetes.csv")
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

val_list = ['Pregnancies', 'Glucose', 'BloodPressure']

train_X, test_X, train_y, test_y = train_test_split(df[val_list], df.Outcome, test_size=0.3, random_state=123)

model = KNeighborsClassifier().fit(train_X, train_y)
```


```python
model.score(test_X, test_y)
```




    0.7272727272727273



답 : 0.73

### Q3. 종속변수를 당뇨 발병 여부로 하고 임신여부, 혈당, 혈압, 인슐린, 체질량지수를 독립변수로 하여 정확도를 확인했을 때 그 k 값과 정확도가 올바르게 연결되지 않은 것은?
- diabates.csv 사용
- 8:2 분할, seed = 123
- k를 제외한 설정은 모두 기본값으로 하시오


```python
df = pd.read_csv("diabetes.csv")
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['is_Pregnancies'] = (df.Pregnancies > 0) + 0
```


```python
val_list = ['is_Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI']

train_X, test_X, train_y, test_y = train_test_split(df[val_list], df.Outcome, test_size=0.2, random_state=123)
```


```python
model = KNeighborsClassifier(n_neighbors=3).fit(train_X, train_y)
model.score(test_X, test_y)
```




    0.7077922077922078




```python
model = KNeighborsClassifier(n_neighbors=5).fit(train_X, train_y)
model.score(test_X, test_y)
```




    0.7337662337662337




```python
model = KNeighborsClassifier(n_neighbors=10).fit(train_X, train_y)
model.score(test_X, test_y)
```




    0.7792207792207793




```python
model = KNeighborsClassifier(n_neighbors=20).fit(train_X, train_y)
model.score(test_X, test_y)
```




    0.7597402597402597



답 : 4번 k = 20, Acc. = 0.79

### Q3. 종속변수를 체질량지수로 하고 임신여부, 혈당, 혈압, 인슐린을 독립변수로 하여 예측값을 확인했을 때 그 k 값과 RMSE가 올바르게 연결되지 않은 것은?
- diabates.csv 파일 사용
- 8:2 분할, seed = 123
- k를 제외한 설정은 모두 기본값으로 하시오.


```python
df = pd.read_csv("diabetes.csv")
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['is_Pregnancies'] = (df.Pregnancies > 0) + 0
```


```python
val_list = ['is_Pregnancies', 'Glucose', 'BloodPressure', 'Insulin']

train_X, test_X, train_y, test_y = train_test_split(df[val_list], df.BMI, test_size=0.2, random_state=123)

model = KNeighborsRegressor(n_neighbors=3).fit(train_X, train_y)
```


```python
from sklearn.metrics import mean_squared_error as mse

mse(y_true = test_y, y_pred=model.predict(test_X)) ** 0.5
```




    8.507554151075245




```python
model = KNeighborsRegressor(n_neighbors=5).fit(train_X, train_y)
mse(y_true = test_y, y_pred=model.predict(test_X)) ** 0.5
```




    8.705726283881784




```python
model = KNeighborsRegressor(n_neighbors=10).fit(train_X, train_y)
mse(y_true = test_y, y_pred=model.predict(test_X)) ** 0.5
```




    8.516665213781529




```python
model = KNeighborsRegressor(n_neighbors=20).fit(train_X, train_y)
mse(y_true = test_y, y_pred=model.predict(test_X)) ** 0.5
```




    8.514480484567331

