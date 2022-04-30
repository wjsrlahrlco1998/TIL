# 1. 딥러닝 모델의 저장과 불러오기



## 1) 모델 저장과 불러오기 간단한 코드



1. 모델 저장

   ```python
   m2.save_weights('m_2_w.h5') # 모델의 가중치를 저장
   m2.save('m_2.h5') # 모델 저장(모델의 형태)
   ```

2. 모델의 구조만 생성하고 가중치 값을 불러와서 사용하기

   ```python
   m3 = keras.Sequential()
   m3.add(keras.layers.Flatten(input_shape=t_x.shape[1:]))
   m3.add(keras.layers.Dense(100, activation='relu', name='h1'))
   m3.add(keras.layers.Dropout(0.3)) # Layer 사이의 뉴런을 30% 제거한다.
   m3.add(keras.layers.Dense(300, activation='relu', name='h2'))
   m3.add(keras.layers.Dense(10, activation='softmax', name='y'))
   m3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
   
   m3.load_weights('m_2_w.h5') # 저장되었던 가중치를 불러온다.
   ```

3. 모델 자체를 불러오기

   ```python
   m4 = keras.models.load_model('m_2.h5')
   ```

   

## 2. 딥러닝 모델의 함수화 + 더 좋은 모델을 생성하기 위한 방법



## 1) 코드

1. import

   ```python
   import tensorflow as tf
   from tensorflow import keras # 신경망 구축을 위한 모듈
   import numpy as np
   from sklearn.model_selection import train_test_split # 데이터 분리를 위한 모듈
   import pandas as pd
   import matplotlib.pyplot as plt
   ```

2. 데이터 로드 및 전처리

   ```python
   # 데이터 로드(학습용, 테스트용) 7:3
   (t_x, t_y), (tt_x, tt_y) = keras.datasets.fashion_mnist.load_data()
   # 데이터 전처리
   s_t_x = t_x / 255.0
   s_tt_x = tt_x/255.0
   # 데이터 분류(학습용, 검증용) 8:2
   t_x, v_x, t_y, v_y = train_test_split(s_t_x, t_y, test_size=0.2, random_state=42)
   ```

3. 모델 생성을 함수화하기

   ```python
   # 모델 생성 함수화 : 정교한 형식도 지정가능
   def my_m(l=None):
       m = keras.Sequential()
       m.add(keras.layers.Flatten(input_shape=(28, 28)))
       if l: # Layer가 있다면 추가
           for i in l:
               m.add(i)
       m.add(keras.layers.Dense(10, activation='softmax'))
       m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
       return m
   ```

4. 학습 중 최상의 파라미터를 가진 모델을 저장

   ```python
   m = keras.Sequential()
   m.add(keras.layers.Flatten(input_shape=(28, 28)))
   m.add(keras.layers.Dense(100, activation='relu'))
   m.add(keras.layers.Dropout(0.3)) # Dropout은 입력단에는 쓰면 안된다.
   m.add(keras.layers.Dense(10, activation='softmax'))
   m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
   
   '''
   save_best_only : best 모델만 저장
   save_weights_only : best 가중치만 저장
   '''
   ck_p = keras.callbacks.ModelCheckpoint('best_m.h', save_best_only=True) # 이상적인 파라미터의 값을 기록
   hy = m.fit(t_x, t_y, epochs=20, validation_data=(v_x, v_y), callbacks=[ck_p])
   plt.plot(hy.history['loss'])
   plt.plot(hy.history['val_loss'])
   plt.show()
   ```

   - keras의 callback을 통해서 학습 중에 이상적인 파라미터 값을 가진 모델을 저장할 수 있다.

5. 학습 중 loss 값이 발산할 때 멈추게하기

   ```python
   m1 = my_m([
       keras.layers.Dense(100, activation='relu'),
       keras.layers.Dropout(0.3) # Dropout은 입력단에는 쓰면 안된다.
   ])
   
   ck_p = keras.callbacks.ModelCheckpoint('best_m.h5', save_best_only=True) # 이상적인 파라미터의 값을 기록
   
   e_st = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True) # 중단지점 설정
   m1.fit(t_x, t_y, epochs=20, validation_data=(v_x, v_y), callbacks=[ck_p, e_st])
   ```

   - EarlyStopping
     - 검증점수가 일정횟수간 개선되지 않으면 Stop한다.
     - EarlyStopping의 중단지점이 반드시 좋은 모델이 되는 것이 아니다.
       - Stop은 검증개선이 되지않았을때이므로 중단 전에 더 좋은 모델이 존재할 수 있다.
     - 파라미터
       - patience : 몇 번 검증점수가 개선되지 않을때 멈출지 설정
       - restore_best_weights : 가장 좋았던 가중치를 저장 -> 모델을 이 가중치로 최종적으로 결정한다.(중단지점의 가중치 아님!!)
         - 이 값은 모델학습 후 곧바로 학습하는것이 아니라면 굳이 설정하지는 않는다.