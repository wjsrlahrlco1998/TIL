# 1. 앙상블 코드 예시(기초)



## 1) 정의

- 여러 분류기들을 모아 하나의 분류기를 만든다
- 입력데이터에 맞는 최적의 분류기의 결과를 도출한다



## 2) 코드

- 데이터 가져오기

  ```python
  import pandas as pd
  import numpy as np
  
  # 1. 데이터 가져오기
  data = pd.read_csv('day6_data1.csv')
  ```

- 분할 속성 추출

  ```python
  # 2. 분할 속성 추출
  X = data[['alcohol', 'sugar', 'pH']].to_numpy()
  Y = data['class'].to_numpy()
  ```

- 사용 데이터 분류

  ```python
  # 3. 사용 데이터 분류 : 반드시 고정되어야하는 데이터 셋
  from sklearn.model_selection import train_test_split
  # 검증 데이터 분리
  t_x, tt_x, t_y, tt_y = train_test_split(X, Y,test_size=0.2, random_state=42)  
  ```

- 앙상블 - 교차검증

  ```python
  # 4. 앙상블 - 교차검증
  from sklearn.model_selection import cross_validate
  from sklearn.ensemble import RandomForestClassifier
  
  rf = RandomForestClassifier(random_state=42, n_jobs=-1) # 랜덤 포레스트 모델 생성(*앙상블)
  sc = cross_validate(rf, t_x, t_y, return_train_score=True, n_jobs=-1) # 교차검증
  np.mean(sc['train_score']), np.mean(sc['test_score']) # 테스트 결과 점수
  
  # 4-1. 직접 학습의 점수 확인
  rf.fit(t_x, t_y)
  rf.feature_importances_ # 각 피쳐의 중요도 측정값 반환
  rf.score(t_x, t_y), rf.score(tt_x, tt_y)
  
  rf1 = RandomForestClassifier(oob_score=True , random_state=42, n_jobs=-1) # 데이터의 특성을 자동으로 처리
  rf1.fit(t_x, t_y)
  rf1.score(t_x, t_y), rf.score(tt_x, tt_y)
  rf1.oob_score_ # 특정한 샘플의 결과 점수 값
  
  # 트리 구조의 앙상블
  from sklearn.ensemble import ExtraTreesClassifier
  
  et = ExtraTreesClassifier(random_state=42, n_jobs=-1) # 모델 생성
  sc = cross_validate(et, t_x, t_y, return_train_score=True, n_jobs=-1) # 교차검증
  np.mean(sc['train_score']), np.mean(sc['test_score']) # 테스트 결과 점수
  
  et.fit(t_x, t_y)
  rf.feature_importances_, et.feature_importances_ # 앙상블 모델별 각 피쳐 중요도 출력
  
  # 부스팅 기법 : 하나의 모델에서 여러 데이터를 샘플링한 다음 그 샘플링된 데이터로 각각의 모델을 만드는 기법
  from sklearn.ensemble import GradientBoostingClassifier # 오차율을 줄여주는 기법
  
  gd = GradientBoostingClassifier(random_state=42)
  sc = cross_validate(gd, t_x, t_y, return_train_score=True, n_jobs=-1) # 교차검증
  np.mean(sc['train_score']), np.mean(sc['test_score'])
  
  # n_estimators : 트리의 개수 지정
  gd = GradientBoostingClassifier(random_state=42, n_estimators=500, learning_rate=0.2)
  sc = cross_validate(gd, t_x, t_y, return_train_score=True, n_jobs=-1) # 교차검증
  np.mean(sc['train_score']), np.mean(sc['test_score'])
  ```

  

