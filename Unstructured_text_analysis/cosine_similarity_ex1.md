# 코사인 유사도를 이용한 영화추천 예시

## * 유사도란?

- 특정 기준을 따라 유사한지의 의미가 달라진다.
- 수학적으로 유사성을 판단하기위해 공간에 좌표로 표현한다.
- 벡터로 유사판단하는 것은 방향성의 유사성이다.
- 데이터의 특성에 따라 유사도 측정 방법이 달라진다.



---

## 코드 예시

---

1. 패키지 import

   ```python
   import pandas as pd
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics.pairwise import cosine_similarity
   ```

2. 데이터 로드 및 결측치 처리

   ```python
   data = pd.read_csv('data1.csv', low_memory=False)
   data = data.head(20000)
   data['overview'] = data['overview'].fillna('') # 결측치 채우기
   ```

3. TF-IDF : 벡터화

   ```python
   tfidf = TfidfVectorizer(stop_words='english') # 불용어 제거
   tfidf_table = tfidf.fit_transform(data['overview'])
   ```

4. cos 계산

   ```python
   cos = cosine_similarity(tfidf_table, tfidf_table)
   ```

5. (key, value)가 (제목, 인덱스)인 dict 생성

   ```python
   t_idx = dict(zip(data['title'], data.index))
   ```

6. 추천 함수

   ```python
   def ck_s_t(title, cosine_sim=cos):
       idx = t_idx[title]
       # enumerate : (인덱스, 값)으로 가져옴
       c_sc = list(enumerate(cosine_sim[idx])) # title에 대한 모든 문서의 유사도
       c_sc = sorted(c_sc, key=lambda x : x[1], reverse=True) # 내림차순 정렬
       m_i = c_sc[1:6] # 자신과 비교한 유사도를 제외하고 유사도가 높은 5개를 가져옴
       m_idx = [i[0] for i in m_i]
       return data['title'].iloc[m_idx]
   ```

7. 결과

   ```python
   ck_s_t('Star Wars')
   ```

   ```
   1154    The Empire Strikes Back
   1167         Return of the Jedi
   1267               Mad Dog Time
   5187        The Triumph of Love
   309           The Swan Princess
   Name: title, dtype: objec
   ```

8. 응용
   - cos.shape을 살펴보면 20000,20000 의 구조를 가지는데 이는 이미지를 분류하는데 이용했던 CNN 층에 적용할 수 있다.