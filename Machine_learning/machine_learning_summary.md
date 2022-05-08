# 1.  머신러닝의 개요(예시)

- 코드

  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn.neighbors import KNeighborsClassifier
  
  #1. 데이터 수집
  data = pd.read_csv('data.csv')
  X = pd.DataFrame(data, columns=['D_length', 'D_weight'])
  Y = pd.DataFrame(data, columns=['y'])
  np_X = np.array(X)
  np_Y = np.array(Y['y'], dtype=int)
  
  #2. 데이터 전처리
  mean = np.mean(np_X, axis=0)
  std = np.std(np_X, axis=0)
  sc_t_X = (np_X - mean) / std
  t_x, tt_x, t_y, tt_y = train_test_split(sc_t_X, np_Y, random_state=10) # 트레이닝 Data와 Test Data 분리
  
  #3. 모델 생성 및 학습
  # KNeighborsClassifier의 옵션 : n_neighbors의 기본 값은 5 
  kn = KNeighborsClassifier().fit(t_x, t_y)
  
  #4. 테스트 및 검증
  kn.score(tt_x, tt_y)
  ```

- 주의사항

  - 데이터의 전처리 시 각 값을 정규화해야 정확한 학습이 진행될 수 있다.
  - Training Data와 Test Data를 분리하여 Test Data를 통해서 테스트 및 검증한다.
  - 머신러닝에서 가장 중요한 것은 데이터의 적절한 전처리로 정확한 결과를 이끌어내는것이다.