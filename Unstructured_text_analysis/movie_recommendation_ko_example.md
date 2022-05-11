# 유사도 : 영화 3종 추천 예시

---

## 1. 데이터 로드

```python
import pandas as pd

original_data = pd.read_csv('data2.csv', encoding='utf-8')
original_data
```

![image-20220511202800915](movie_recommendation_ko_example.assets/image-20220511202800915.png)

```python
# 결측치 점검
original_data.isnull().sum()

# 결측치 채우기
original_data['genre'] = original_data['genre'].fillna('')
original_data['genre']
```



## 2. 데이터 전처리 : 형태소 토큰화 및 TD-IDF 벡터화

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt

ok = Okt()

tf_idf = TfidfVectorizer(tokenizer=ok.morphs)
tf_idf_table = tf_idf.fit_transform(original_data['content'])

tf_idf.vocabulary_
```

![image-20220511202859249](movie_recommendation_ko_example.assets/image-20220511202859249.png)



## 3. 코사인 유사도 기반 영화 3종 추천

```python
from sklearn.metrics.pairwise import cosine_similarity

# 코사인 유사도 계산
cos = cosine_similarity(tf_idf_table, tf_idf_table)

# 제목 : index 형식의 dict 생성
title_idx = dict(zip(original_data['name'], original_data.index))
title_idx
```

![image-20220511202928129](movie_recommendation_ko_example.assets/image-20220511202928129.png)

```python
def cos_sim(title, cosine_sim=cos):
    idx = title_idx[title]
    cosin_score = list(enumerate(cosine_sim[idx]))
    cosin_score = sorted(cosin_score, key = lambda x : x[1], reverse = True)
    high_score_3 = cosin_score[1:4]
    high_score_3_idx = [i[0] for i in high_score_3]
    return original_data['name'].iloc[high_score_3_idx]

cos_sim('올드보이')
```

```
11    어바웃 타임
5     친절한금자씨
16     인터스텔라
Name: name, dtype: object
```

---

## * 유클리드 유사도 이용

```python
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

e_tf_idf = tf_idf_table / np.sum(tf_idf_table)
euc = euclidean_distances(e_tf_idf, e_tf_idf)

def euc_sim(title, euc_sim=euc):
    idx = title_idx[title]
    euc_score = list(enumerate(euc_sim[idx]))
    euc_score = sorted(euc_score, key = lambda x : x[1], reverse = False)
    high_score_3 = euc_score[1:4]
    high_score_3_idx = [i[0] for i in high_score_3]
    return original_data['name'].iloc[high_score_3_idx]

euc_sim('올드보이')
```

```
11    어바웃 타임
5     친절한금자씨
16     인터스텔라
Name: name, dtype: object
```

---

## * 맨허튼 유사도 이용

```python
from sklearn.metrics.pairwise import manhattan_distances

e_tf_idf = tf_idf_table / np.sum(tf_idf_table)
mht = euclidean_distances(e_tf_idf, e_tf_idf)

def mht_sim(title, mht_sim=mht):
    idx = title_idx[title]
    mht_score = list(enumerate(mht_sim[idx]))
    mht_score = sorted(mht_score, key = lambda x : x[1], reverse = False)
    high_score_3 = mht_score[1:4]
    high_score_3_idx = [i[0] for i in high_score_3]
    return original_data['name'].iloc[high_score_3_idx]

mht_sim('올드보이')
```

```
11    어바웃 타임
5     친절한금자씨
16     인터스텔라
Name: name, dtype: object
```

---

## * 형태소 분석을 다르게 한 결과

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt

okt = Okt()
data_l = [' '.join(okt.morphs(i)) for i in data['content']]

tfidf_v = TfidfVectorizer()
tfidf_m = tfidf_v.fit_transform(data_l)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import euclidean_distances
cos=cosine_similarity(tfidf_m,tfidf_m)#숫자가 높으면 유사도는?->높다
c1=manhattan_distances(tfidf_m,tfidf_m)#숫자가 높으면 유사도는?->낮다
c2=euclidean_distances(tfidf_m,tfidf_m)

t_idx=dict(zip(data['name'],data.index))

def ck_s_t(t,cosine_sim=cos,k=0):
    idx=t_idx[t]
    c_sc=list(enumerate(cosine_sim[idx]))
    if k==0:
        c_sc=sorted(c_sc,key=lambda x:x[1],reverse=True)#내림차순 정렬
    else:
        c_sc=sorted(c_sc,key=lambda x:x[1])
    m_i=c_sc[1:4]
    m_idx=[i[0] for i in m_i]
    return data['name'].iloc[m_idx]

ck_s_t('올드보이',k=0)
ck_s_t('올드보이',c1,k=1)
ck_s_t('올드보이',c2,k=1) 
```

```
5     친절한금자씨
11    어바웃 타임
8        아저씨
Name: name, dtype: object
```

```
10       장고
1       노트북
13    트랜스포머
Name: name, dtype: object
```

```
5     친절한금자씨
11    어바웃 타임
8        아저씨
Name: name, dtype: object
```