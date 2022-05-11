# 가설 검정과 t-검정



### 1. 가설 검정

- 귀무가설(기존의 주장) 정의
  - ex. 마포구와 영등포구의 공유자전거 이동거리의 평균은 같다.
- 대립가설
  - ex. 작다, 크다, 같지않다
- 유의수준과 P-value 간의 비교
  - 유의수준 0.05 : 통계적으로 의미 있는 차이의 기준값
  - P-value : 귀무가설이 참이라는 가정 하에 현재 두 평균값 사이의 차이가 나타날 확률
  - P-value < 0.05이면 귀무가설을 기각
  - 유의수준은 고정된 기준값이 아닌, 해결해야할 문제에 따라 값이 달라진다. 
  - 오차에 민감할수록 유의수준의 값을 낮춘다.



### 2. t-검정

- One sample t-test
  - 한쪽은 알려진 값으로 평균값이 제시되고, 다른 한쪽은 데이터로 제시되어 평균을 구함
  - 알려진 평균값과 계산한 표본 평균을 비교함
- Paired t-test
  - ex. 영어 성적 10점 향상 공부법
  - 해당 학습법으로 공부하기 이전과 이후의 점수 변화 비교
- Two sample t-test



---

### 2개의 구의 자전거 이동거리 평균을 비교

---

1. 영등포구와 마포구의 데이터 필터링

   ```python
   y_gu = bike_data2[bike_data2.Gu == '영등포구']
   m_gu = bike_data2[bike_data2.Gu == '마포구']
   ```

2. Levene 등분산 검정을 활용하여 등분산 판단

   ```python
   from scipy import stats
   
   stats.levene(y_gu.Distance, m_gu.Distance)
   ```

   ```tex
   LeveneResult(statistic=3.5647234607192013, pvalue=0.05903430224682354)
   ```

​		p-value > 0.05이므로 등분산이다.

3. 평균 비교

   ```python
   np.mean(y_gu.Distance), np.mean(m_gu.Distance)
   ```

   ```
   4190.278911564626 4514.426966292135
   ```

4. 평균의 차이가 일시적인것인지 아닌지 확인

   ```python
   stats.ttest_ind(y_gu.Distance, m_gu.Distance, equal_var = True)
   ```

   - 등분산이므로 ``equal_var = True``

   ```
   Ttest_indResult(statistic=-4.002195758414915, pvalue=6.298774059911862e-05)
   ```

​		p-value가 0.05보다 작음 -> 귀무가설 기각 == 두 구의 이동거리 평균은 같지않다.