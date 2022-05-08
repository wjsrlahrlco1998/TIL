# 1. 분류모델의 성능평가

## 1) 혼동행렬

### (1) 정의

- 예측값이 실제값 대비 얼마나 잘 맞는지 2X2 행렬로 표현

### (2) 모델 평가 성능지표

- 분류
  - 정확도, 정밀도, 민감도, F1 스코어, ROC 커브, 리프트 차트

### (3) 코드

- 혼동행렬

  ```python
  # 혼동행렬의 성능 판단은 이진 분류를 베이스로 성능을 판단한다
  from sklearn.metrics import confusion_matrix # 혼동행렬을 사용하기 위한 클래스
  y_t = [1, 0, 1, 1, 0, 1] # 실제 결과값
  y_p = [0, 0, 1, 1, 0, 1] # 예측 결과값
  
  confusion_matrix(y_t, y_p) # 2X2
  
  # 혼동행렬의 원소를 각각에 대입
  tn, fp, tp, fn = confusion_matrix(y_t, y_p).ravel() 
  ```

  

- 정확도

  -  (TP + TN) / (TP + TN + FP + FN) : (정답을 맞춘 수) / (전체 개수)

    ```python
    import numpy as np
    
    y_t = [1, 0, 1, 1, 0, 1] # 실제 결과값
    y_p = [0, 0, 1, 1, 0, 1] # 예측 결과값
    
    # 연산을 위해 Numpy 배열로 만든다
    y_t = np.array(y_t)
    y_p = np.array(y_p)
    
    sum(y_t == y_p) / len(y_t)
    ```

  - 사이킷런 내부 클래스 사용

    ```python
    from sklearn.metrics import accuracy_score
    
    accuracy_score(y_t, y_p) # 이미 존재하는 메서드를 이용
    ```

- 정밀도

  - TP / (TP + FP) : (예측의 정답 수) / (전체 예측 값의 수)

    ```python
    # 불균일한 데이터를 다룰 때 유용
    from sklearn.metrics import precision_score
    
    precision_score(y_t, y_p)
    ```

- 민감도

  - TP / (TP + FN) : 실제 1인 값을 가진 데이터의 모델이 얼마나 1이라고 잘 예측했는지에 대한 비율

    ```python
    from sklearn.metrics import recall_score
    
    recall_score(y_t, y_p)
    ```

- F1 스코어

  - 정밀도와 민감도의 조화평균 값

    ```python
    from sklearn.metrics import f1_score
    
    f1_score(y_t, y_p)
    ```



# 2. 모델 성능 평가 예시

## 1) 코드

1. Data 수집

   ```python
   import pandas as pd
   
   # 데이터 Load
   df = pd.read_csv('day5_data1.csv')
   ```

2. 데이터 전처리

   - 필요유무에 따른 정리

     ```python
     # 필요없는데이터 제거
     df.pop('who')
     df.pop('Country')
     df.pop('Years on Internet')
     ```

   - 범주화 : 원-핫-인코딩

     ```python
     # 범주화 : 원 핫 인코딩
     ck_c = ['Gender', 'Household Income', 'Sexual Preference', 'Education Attainment', 'Major Occupation', 'Marital Status']
     
     for i in ck_c:
         df[i] = df[i].astype('category') # (1)astype : 형변환 / catecory : 범주형 데이터 -> 자동적으로 원 핫 인코딩이 적용된다.
         
     df_one_hot = pd.get_dummies(df) # (2) 원 핫 인코딩 : category 형 데이터만 원 핫 인코딩 한다
     ```

   - 결측치 처리

     ```python
     # 피쳐마다의 결측치의 수 확인
     df_one_hot.isnull().sum()
     
     # NaN 영역만 선택해서 값을 바꾸기 : 평균값을 채우는 방법 사용
     df_one_hot.loc[pd.isnull(df_one_hot['Age']), 'Age'] = df_one_hot['Age'].mean()
     ```

   - 입력 Data 정리

     ```python
     X = df_one_hot.iloc[:, 1:].values # 입력 데이터 지정 
     Y = df_one_hot.iloc[:, 0].values.reshape(-1, 1) # 출력 데이터 지정(결과)
     print(X.shape, Y.shape) # 입력, 출력 데이터의 형태 확인
     ```

   - 데이터 스케일링(정규화)

     ```python
     from sklearn.preprocessing import MinMaxScaler
     m_m_s = MinMaxScaler()
     X_data = m_m_s.fit_transform(X) # 학습과 transform 동시에 진행하는 메서드
     ```

   - 데이터 분류(학습용과 테스트용)

     ```python
     from sklearn.model_selection import train_test_split
     t_x, tt_x, t_y, tt_y = train_test_split(X_data, Y, test_size=0.3, random_state=42)
     ```

3. 모델 생성 및 학습

   ```python
   # 로지스틱 회귀
   from sklearn.linear_model import LogisticRegression
   lo_g = LogisticRegression(fit_intercept=True) # 모델 생성 : fit_intercept : 모형의 상수항을 결정
   lo_g.fit(t_x, t_y.flatten()) # 모델 학습
   ```

4. 테스트 및 검증

   ```python
   # 직접 비교
   lo_g.predict(tt_x[:5])
   tt_y.flatten()[:5]
   
   lo_g.predict_proba(tt_x[:5]) # 확률 값 확인
   
   from sklearn.metrics import confusion_matrix # 혼동행렬 사용
   from sklearn.metrics import accuracy_score # 정확도 확인
   
   y_t = tt_y.copy() # 정답값
   y_p = lo_g.predict(tt_x) # 예측값
   confusion_matrix(y_t, y_p)
   
   accuracy_score(y_t, y_p) # 정확도 평가
   
   # 성능에 대한 종합 정보(정확도, 정밀도, f1-score)를 확인하는 메서드 사용
   from sklearn.metrics import classification_report
   
   print(classification_report(y_t, y_p))
   ```

   * 분류 모델의 성능 검증은 비율로 확인해야 정확하다.



# 3. Soft Max 함수를 활용한 다중 분류기

## 1) 다중 분류

- 정의

  - 2개 이상의 클래스를 가진  y 값에 대한 분류

- 종류

  - 다중클래스 분류 
    - 중복 선택이 불가능한 분류
  - 다중레이블 분류
    - 중복 선택이 가능한 분류

- 분류방법

  - One-vs-All

    - m개의 클래스가 존재할 때 각 클래스마다 분류기를 생성하여 분류

    - One-vs-Rest

    - 대표적으로 소프트맥스 분류

    - 형식

      ```
      ovr => [a, b, c]
      a! = [b, c]
      b! = [a, c]
      c! = [a, b]
      ```

  - One-vs-One

    - m개의 클래스가 있을 때, 이 클래스의 분류기를 하나의 클래스로 하고 나머지 클래스의 분류기를 만들어 최종적으로 각 분류기들의 결과를 투표로 결정

    - 분류기가 많아질수록 정확도도 높아지지만 비용도 증가한다

    - 형식

      ```
      ovo => [a, b, c]
      a! = b
      a! = c
      b! = a
      b! = c
      c! = a
      c! = b
      ```

- Soft Max 수식코드

  ```python
  # softmax 수식 코드로 작성하기
  import numpy as np
  
  # soft_max 함수 작성
  def s_max(z):
      a_v = np.exp(z)
      return a_v / sum(a_v)
  
  z = [2, 1, 5, 0, 5]
  y = s_max(z)
  ```



## 2) 다중 분류 예시

- 코드

  1. 데이터 수집

     ```python
     # minist 데이터셋 이용 -> 데이터 셋은 이미 전처리가 완료되어있다
     from sklearn.datasets import load_digits
     data = load_digits()
     
     data['data'][0].shape # 피쳐의 개수 확인
     ```

  2. 데이터 전처리

     ```python
     # 입력값과 결과값 지정
     X = data['data']
     Y = data['target']
     
     # 데이터 분류
     from sklearn.model_selection import train_test_split
     
     t_x, tt_x, t_y, tt_y = train_test_split(X, Y, random_state=42)
     ```

  3. 모델 생성 및 학습

     ```python
     # 모델 생성 및 학습
     from sklearn.linear_model import LogisticRegression
     
     # orv : 각각의 클래스마다 이진분류 형태로 데이터 구축
     lo_g_ovr = LogisticRegression(multi_class='ovr') # 다중 분류 형태 지정
     
     # multinomial : soft_max 함수를 사용하여 계산 - gradient decent 함수 사용
     # solver : 매개변수 제시(soft_max 를 사용하기 위해서는 전달해야함)
     lo_g_s_max = LogisticRegression(multi_class='multinomial', solver='sag')
     
     # 학습
     lo_g_ovr.fit(t_x, t_y) 
     lo_g_s_max.fit(t_x, t_y)
     ```

  4. 모델 검증 및 평가

     ```python
     # 모델 검증 및 평가
     from sklearn.metrics import confusion_matrix
     
     y_t = tt_y.copy()
     y_p = lo_g_ovr.predict(tt_x)
     
     confusion_matrix(y_t, y_p)
     
     # 성능에 대한 종합 정보(정확도, 정밀도, f1-score)를 확인하는 메서드 사용
     from sklearn.metrics import classification_report
     
     print(classification_report(y_t, y_p))
     ```



# 4. ROC 커브

- 정의

  - 분류기의 임계값을 지속적으로 조정하여 정밀도와 민감도 간의 비율을 도식화
  - 클래스의 예측 확률이 나오는 모델에 사용이 가능

- AUC(Area Under Curve)

  - ROC 커브 하단의 넓이
  - 불균형 데이터셋의 성능을 평가할 때 사용

- 코드

  ```python
  # roc_curve Test
  import numpy as np
  from sklearn.metrics import roc_curve
  
  # roc curve를 확인하기 위한 임의의 값
  y = np.array([1, 1, 2, 2])
  sc = np.array([0.1, 0.4, 0.35, 0.8])
  fpr, tpr, th = roc_curve(y, sc, pos_label = 2) # roc_curve(정답, 확률) / pos_label : 확인할 항목
  print(fpr, tpr, th)
  
  # auc (roc 커브 하단의 면적) 확인
  from sklearn.metrics import auc
  roc_auc = auc(fpr, tpr)
  roc_auc
  ```

  