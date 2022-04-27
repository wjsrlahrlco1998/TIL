# 1. 앙상블



## 1) 정의

- 데이터 분석 알고리즘으로 여러 개의 알고리즘을 사용하여 예측된 결과들을 투표(다수결)에 의해서 최종 결과 값을 도출하는 기법이다.



## 2) 여러 기법

- 바닐라 앙상블
  - 기본적인 앙상블 기법으로, 아무것도 처리하지않은 앙상블 기법을 말한다.
  - 일반적으로 가중치 평균이나 투표 방식으로 만들어지는 앙상블 모델이다.
- 부스팅
  - 학습을 단계별로 차례로 진행하면서 각 예측이 틀린 데이터에 가중치를 주는 방식
  - 최종적으로 잘못 분류된 데이터를 좀 더 잘 분류할 수 있는 모델로 만들어준다
  - 높은 성능이 필요할 때 사용하는 모델이다
- 배깅
  - 데이터 셋에서 N 개의 부분 집합을 추출하여 하나의 모델에 다양한 데이터를 넣어서 N개의 모델을 생성하여, 각 모델들의 예측치를 투표하여 최종적인 예측치를 도출하는 기법이다



## 3) 앙상블 코드 예시



### (1) 앙상블 - 투표 분류기(바닐라모델)

1. import

   ```python
   import numpy as np
   from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모듈
   from sklearn.tree import DecisionTreeClassifier # 의사결정트리(분류) 모듈
   from sklearn.naive_bayes import GaussianNB # 나이브 베이지안 : 가우시안 모듈
   from sklearn.ensemble import VotingClassifier # 앙상블 - 투표분류기 (분류 모듈들을 묶음)
   ```

2. 데이터 Load

   ```python
   # np.load() : npy 형태의 파일을 읽어온다
   # 미리 처리된 형태
   X = np.load('day7_data2_X.npy') # npy 데이터 파일 가져오기
   Y = np.load('day7_data2_Y.npy')
   ```

3. 앙상블 모델에 들어갈 분류기 모델 생성

   ```python
   c1 = LogisticRegression(random_state=1) # 분류모델 1 : 로지스틱
   c2 = DecisionTreeClassifier(random_state=1, max_depth=4) # 분류모델 2 : 결정트리
   c3 = GaussianNB() # 분류모델 3 : 나이브 베이지안 가우시안
   ```

4. 각 분류기들을 투표 분류기에 묶음(앙상블)

   ```python
   # 객체 전달 형태 : [('이름', 객체), ...]
   # voting : 
   ec = VotingClassifier(estimators=[('lr', c1), ('rf', c2), ('dt', c3)], voting='hard')
   ```

5. 교차 검증을 통한 성능 비교

   (1) 앙상블된 모델의 성능 교차검증

   ```python
   from sklearn.model_selection import cross_val_score # 교차검증의 점수 확인을 위한 모듈
   
   # 매개변수 ; (검증할 모델, X, Y)
   # cv : 교차 검증의 반복 횟수 지정
   # (1) 앙상블된 모델의 성능 교차검증
   cross_val_score(ec, X, Y, cv=5).mean() 
   ```

   ​	실행결과 : 0.8222941661905668

   (2) 로지스틱 회귀 모델의 단일 교차검증

   ```python
   # 로지스틱 단일 검증
   cross_val_score(c1, X, Y, cv=5).mean() 
   ```

   ​	실행결과 : 0.8290420872214816

   (3) 결정트리 분류 모델의 단일 검증

   ```python
   # 결정트리 단일 검증
   cross_val_score(c2, X, Y, cv=5).mean() 
   ```

   ​	실행결과 : 0.8223068621849807

   (4) 가우시안 모델의 단일 검증

   ```python
   # 가우시안 단일 검증
   cross_val_score(c3, X, Y, cv=5).mean()
   ```

   ​	실행결과 : 0.4600139655938551

   (5) 성능 비교 분석

   - 가우시안 모델의 성능이 현저히 낮기 때문에 제외하여 앙상블한다. -> 앙상블 모델의 성능 개선

6. 성능이 떨어지는 모델 제외 후 재 앙상블

   ```python
   ec1 = VotingClassifier(estimators=[('lr', c1), ('dt', c2)], voting='hard')
   cross_val_score(ec1, X, Y, cv=5).mean() 
   ```

   실행결과 : 0.8301783787215135

   - 성능이 조금 더 개선되었다

7. 성능 개선을 위한 하이퍼 파라미터 값 조정

   ```python
   c_params = [0.1, 5.0, 7.0, 10.0, 15.0, 20.0, 100.0] # 파라미터 값 설정
   
   # 파라미터 값 제시 // 값을 리스트로 주는 이유는 extend 형식으로 받기 때문이다
   # 앙상블에서 설정한 이름__ 으로 어떤 모델의 파라미터 값을 지칭하는것인지 표현
   params = { 
       'lr__solver' : ['liblinear'], # solver : 
       'lr__penalty' : ['l2'], # penalty :
       'lr__C' : c_params, # C : 
       'dt__criterion' : ['gini', 'entropy'], # criterion : gain의 계산방법을 지니계수, 엔트로피 중 선택한다
       'dt__min_samples_leaf' : [10, 8, 7, 6, 5, 4, 3, 2], # min_samples_leaf : 
       'dt__max_depth' : [1, 2, 3, 4, 5, 6, 7, 8, 9] # max_depth : 트리의 최대 깊이를 지정
   }
   
   from sklearn.model_selection import GridSearchCV # 최적의 파라미터를 찾기위한 모듈
   
   g = GridSearchCV(estimator=ec1, param_grid=params, cv=5)
   g.fit(X, Y)
   g.best_score_ # 최고 점수 값 확인
   ```

   실행결과 : 0.8436869167777565

   - 하이퍼 파라미터 값 조정을 통해서 성능을 조금 더 상승시켰다

8. 최적의 하이퍼 파라미터를 적용한 앙상블 모델

   ```python
   # 그리드 서치로 다시 적용 -> 통상적이지 않은 방법 그냥 테스트
   ec2 = VotingClassifier(estimators=[('A', c1), ('B', c2)], voting='hard')
   params = {'B__criterion': ['gini'],
    'B__max_depth': [9],
    'B__min_samples_leaf': [5],
    'A__C': [5.0],
    'A__penalty': ['l2'],
    'A__solver': ['liblinear']}
   
   g1 = GridSearchCV(estimator=ec2, param_grid=params, cv=5)
   
   # 최적의 하이퍼 파라미터 값 적용
   c1 = LogisticRegression(solver='liblinear', penalty= 'l2', C=5.0, random_state=1) # 분류모델 1 : 로지스틱
   c2 = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=5, random_state=1) # 분류모델 2 : 결정트리
   
   ec3 = VotingClassifier(estimators=[('A', c1), ('B', c2)], voting='hard')
   cross_val_score(ec3, X, Y, cv=5).mean() # 점수값
   ```



### (2) 배깅 모델

- 랜덤포레스트 모델은 배깅기법과 결정트리 모델을 합친 모델이다.

- 코드

  1. Import

     ```python
     import numpy as np
     from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모듈
     from sklearn.tree import DecisionTreeClassifier # 의사결정트리(분류) 모듈
     from sklearn.naive_bayes import GaussianNB # 나이브 베이지안 : 가우시안 모듈
     from sklearn.ensemble import VotingClassifier # 앙상블 - 투표분류기 (분류 모듈들을 묶음)
     from sklearn.ensemble import BaggingClassifier # 앙상블 - 배깅을 위한 모듈
     from sklearn.model_selection import cross_val_score # 교차검증의 점수 확인을 위한 모듈(성능평가)
     ```

  2. 데이터 Load

     ```python
     # 미리 처리된 형태
     X = np.load('day7_data2_X.npy') # npy 데이터 파일 가져오기
     Y = np.load('day7_data2_Y.npy')
     ```

  3. 배깅 모델 생성

     ```python
     # 배깅 모델 생성 시 사용할 객체를 제시해야 한다
     # oob_score=True : oob 점수 산출 여부 결정
     ec1 = BaggingClassifier(LogisticRegression(random_state=1), oob_score=True)
     ```

  4. 배깅 모델의 교차검증 평가

     ```python
     cross_val_score(ec1, X, Y, cv=5).mean()
     ```

     실행결과 : 0.8166825366596839

  5. 최적의 하이퍼 파라미터를 찾기

     ```python
     from sklearn.model_selection import GridSearchCV # 최적 파라미터 값을 찾기위한 모듈
     
     param = {
         'n_estimators' : [10, 20, 30, 40, 50, 55], # subset으로 생성되는 모델의 개수
         'max_samples' : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 샘플 비율 조정
     }
     
     g = GridSearchCV(ec1, param, cv=5)
     g = g.fit(X, Y) # 학습의 결과물을 g로 받아야 한다
     g.best_params_ # 최적의 파라미터
     ```

     실행결과 : {'max_samples': 0.7, 'n_estimators': 40}



### (3) 부스팅 모델

1. Import

   ```python
   import numpy as np
   from sklearn.model_selection import cross_val_score
   from sklearn.tree import DecisionTreeClassifier
   ```

2. 데이터 Load

   ```python
   X = np.load('day7_data2_X.npy') # npy 데이터 파일 가져오기
   Y = np.load('day7_data2_Y.npy')
   ```

3. 부스팅 모델 생성

   ```python
   from sklearn.ensemble import AdaBoostClassifier # Ada 부스팅 모델
   '''
   base_estimator : 베이스 모델 지정
   n_estimators : 생성할 모델의 수 지정
   '''
   aec = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),# max_depth를 얕게하면 더 적은 정보를 가진다
                           n_estimators=500)
   ```

4. 부스팅 모델의 교차검증

   ```python
   cross_val_score(aec, X, Y, cv=5).mean()
   ```

   실행결과 : 0.808817368120358

5. 배깅 모델과의 성능 비교

   **비교 시에 생성 모델의 수는 같게 지정한다**

   ```python
   from sklearn.ensemble import RandomForestClassifier
   
   ck1 = RandomForestClassifier(n_estimators=500)
   cross_val_score(ck1, X, Y, cv=5).mean()
   ```

   실행결과 : 0.8009395035866185

   - 근소한 차이를 보이지만 부스팅 모델의 성능이 조금 더 높게 나옴을 알 수 있다.

6. 하이퍼 파라미터 조정

   ```python
   aec1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=500)
   # 파라미터 값 지정
   params = {
       'n_estimators' : [20, 22, 23, 24, 25], # 생성할 모델의 수 지정
       'learning_rate' : [0.4, 0.45, 0.5, 0.55, 0.6], # 학습률 지정
       'base_estimator__criterion' : ['gini', 'entropy'], # 트리의 학습방법 설정(엔트로피 or 지니) // 중요한 개념이니 꼭 기억하자
       'base_estimator__max_features' : [7, 8], # 최대 피처 지정(데이터가 가진 피처 개수를 넘어서는 안된다)
       'base_estimator__max_depth' : [1, 2] # 부스팅 할 때 트리의 깊이는 얕게한다
   }
   
   # 그리드 서치를 통해서 최적의 하이퍼 파라미터 찾기
   from sklearn.model_selection import GridSearchCV
   g = GridSearchCV(estimator=aec1, param_grid=params, cv=5, n_jobs=-1)
   g.fit(X, Y)
   g.best_score_
   ```

   실행결과 : 0.8313146702215451



### (4) Gradient Boosting 모델

1. import

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.ensemble import ExtraTreesClassifier
   ```

2. 데이터 Load

   ```python
   # 1. 데이터 가져오기
   data = pd.read_csv('day6_data1.csv')
   
   # 2. 분할 속성 추출
   X = data[['alcohol', 'sugar', 'pH']].to_numpy()
   Y = data['class'].to_numpy()
   
   # 3. 사용 데이터 분류 : 반드시 고정되어야하는 데이터 셋
   from sklearn.model_selection import train_test_split
   # 검증 데이터 분리
   t_x, tt_x, t_y, tt_y = train_test_split(X, Y,test_size=0.2, random_state=42)  
   ```

3. 트리 구조의 앙상블 모델 생성

   ```python
   # 트리 구조의 앙상블
   from sklearn.ensemble import ExtraTreesClassifier
   
   et = ExtraTreesClassifier(random_state=42, n_jobs=-1) # 모델 생성
   sc = cross_validate(et, t_x, t_y, return_train_score=True, n_jobs=-1) # 교차검증
   np.mean(sc['train_score']), np.mean(sc['test_score']) # 테스트 결과 점수
   
   et.fit(t_x, t_y)
   rf.feature_importances_, et.feature_importances_ # 앙상블 모델별 각 피쳐 중요도 출력
   ```

4. Gradient Boosting 모델 생성

   ```python
   from sklearn.ensemble import GradientBoostingClassifier # 오차율을 줄여주는 기법
   
   gd = GradientBoostingClassifier(random_state=42)
   sc = cross_validate(gd, t_x, t_y, return_train_score=True, n_jobs=-1) # 교차검증
   np.mean(sc['train_score']), np.mean(sc['test_score'])
   ```

5. 하이퍼 파라미터 값 조정

   ```python
   # n_estimators : 트리의 개수 지정
   gd = GradientBoostingClassifier(random_state=42, n_estimators=500, learning_rate=0.2)
   sc = cross_validate(gd, t_x, t_y, return_train_score=True, n_jobs=-1) # 교차검증
   np.mean(sc['train_score']), np.mean(sc['test_score'])
   ```

6. Test

   ```python
   from sklearn.experimental import enable_hist_gradient_boosting
   from sklearn.ensemble import HistGradientBoostingClassifier
   
   h = HistGradientBoostingClassifier(random_state=42)
   sc = cross_validate(h, t_x, t_y, return_train_score=True, n_jobs=-1)
   np.mean(sc['train_score']), np.mean(sc['test_score'])
   ```

   

