# 한국어 키버트(Korean KeyBERT)를 이용한 키워드 추출



## 1. 필요 패키지 설치

```python
!pip install sentence_transformers
!pip install konlpy
```



## 2. 간단한 KeyBERT 예시



### 1) 패키지 Load

```python
import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
```



### 2) 사용할 Doc

```python
doc = """
드론 활용 범위도 점차 확대되고 있다. 최근에는 미세먼지 관리에 드론이 활용되고 있다.
서울시는 '미세먼지 계절관리제' 기간인 지난달부터 오는 3월까지 4개월간 드론에 측정장치를 달아 미세먼지 집중 관리를 실시하고 있다.
드론은 산업단지와 사업장 밀집지역을 날아다니며 미세먼지 배출 수치를 점검하고, 현장 모습을 영상으로 담는다.
영상을 통해 미세먼지 방지 시설을 제대로 가동하지 않는 업체와 무허가 시설에 대한 단속이 한층 수월해질 전망이다.
드론 활용에 가장 적극적인 소방청은 광범위하고 복합적인 재난 대응 차원에서 드론과 관련 전문인력 보강을 꾸준히 이어가고 있다.
지난해 말 기준 소방청이 보유한 드론은 총 304대, 드론 조종 자격증을 갖춘 소방대원의 경우 1,860명이다.
이 중 실기평가지도 자격증까지 갖춘 ‘드론 전문가’ 21명도 배치돼 있다.
소방청 관계자는 "소방드론은 재난현장에서 영상정보를 수집, 산악ㆍ수난 사고 시 인명수색·구조활동,
유독가스·폭발사고 시 대원안전 확보 등에 활용된다"며
"향후 화재진압, 인명구조 등에도 드론을 활용하기 위해 연구개발(R&D)을 하고 있다"고 말했다.
"""
```



### 3) 형태소 분석기를 이용하여 명사만 추출한 Doc을 만든다.

```python
okt = Okt()

tokenized_doc = okt.pos(doc)
tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
print('명사 추출 :',tokenized_nouns)
```

```tex
품사 태깅 10개만 출력 : [('\n         ', 'Foreign'), ('드론', 'Noun'), ('활용', 'Noun'), ('범위', 'Noun'), ('도', 'Josa'), ('점차', 'Noun'), ('확대', 'Noun'), ('되고', 'Verb'), ('있다', 'Adjective'), ('.', 'Punctuation')]
명사 추출 : 드론 활용 범위 점차 확대 최근 미세먼지 관리 드론 활용 서울시 미세먼지 계절 관리제 기간 지난달 개 월간 드론 측정 장치 달 미세먼지 집중 관리 실시 드론 산업 단지 사업 밀집 지역 미세먼지 배출 수치 점검 현장 모습 영상 영상 통해 미세먼지 방지 시설 제대로 가동 업체 무허가 시설 대한 단속 한층 전망 드론 활용 가장 적극 소방청 복합 재난 대응 차원 드론 관련 전문 인력 보강 어가 지난해 말 기준 소방청 보유 드론 총 드론 조종 자격증 소방대 경우 명 이 중 실기 평가 지도 자격증 드론 전문가 명도 배치 소방청 관계자 소방 드론 재난 현장 영상 정보 수집 산악 수난 사고 시 인명 수색 구조 활동 유독가스 폭발사고 시 대원 안전 확보 등 활용 며 향후 화재 진압 인명구조 등 드론 활용 위해 연구개발 고 말
```



### 4) CounterVectorizer를 사용하여 단어 추출

- n_gram_range의 인자를 사용하여 n_gram을 추출하기 위함.

```python
n_gram_range = (2, 3)

count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
candidates = count.get_feature_names_out()

print('trigram 개수 :',len(candidates))
print('trigram 다섯개만 출력 :',candidates[:5])
```

```
trigram 개수 : 222
trigram 다섯개만 출력 : ['가동 업체' '가동 업체 무허가' '가장 적극' '가장 적극 소방청' '경우 실기']
```



### 5) 문서와 키워드들을 SBERT를 사용하여 수치화한다.

#### 5-1) 한국어를 포함하는 다국어 SBERT를 로드

```python
# SBERT 로드
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
# 문서 임베딩
doc_embedding = model.encode([doc])
# 키워드 임베딩
candidate_embeddings = model.encode(candidates)
```



### 6) 문서와 가장 유사한 키워드 추출

- 상위 5개만 추출한다.

```python
top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(keywords)
```

```
['드론 산업', '드론 드론 조종', '실시 드론 산업', '관리 드론 활용', '미세먼지 관리 드론']
```



- 결과로 출력된 키워드들은 조금 부자연스러울 수 있다. 이를 해결하기 위해서 다음과 같은 알고리즘을 사용하여 성능을 향상시킬 수 있다.
  - Max Sum Similarity
  - Maximal Marginal Relevance



## 3. Max Sum Similarity

- 데이터 쌍 사이의 최대 합 거리는 데이터 쌍 간의 거리가 최대화 되는 데이터 쌍으로 정의된다.
- 여기서 Max Sum Similarity의 의도는 후보 간의 유사성을 최소화하면서 문서와의 후보 유사성을 극대화하는 것이다.



### 1) Max sum Similarity 함수 정의

```python
def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]
```



### 2) 키워드 선택

- 문서와 유사한 상위 10개의 키워드를 추출한 후, 10개의 키워드 중 가장 덜 유사한 5개의 키워드를 선택한다.

```python
max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10)
```

```
['드론 산업 단지', '전망 드론 활용', '드론 산업', '관리 드론 활용', '미세먼지 관리 드론']
```

- 결과를 보면 단순 코사인 유사도를 사용한 것과 비슷한 결과를 보인다.
- 그렇다면 nr_candidates를 늘린 결과를 확인한다.

```python
max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=30)
```

```
['소방 드론 재난', '자격증 드론 전문가', '월간 드론 측정', '전망 드론 활용', '미세먼지 관리 드론']
```

- 결과적으로 높은 nr_candidates는 더 다양한 키워드를 생성함을 알 수 있다.



## 4. Maximal Marginal Relevance

- 키워드를 다양화하는 마지막 방법.
- 중복을 최소화하고 결과의 다양성을 극대화하는 방법이다.

1. 문서와 가장 유사한 키워드/키프레이즈를 선택한다.
2. 문서와 유사하고 이미 선택된 키워드/키프레이즈와 유사하지 않은 새로운 후보를 반복적으로 선택한다.



### 1) Maximal Marginal Relevance 함수 정의

```python
def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]
```



### 2) 키워드 추출

- 낮은 diversity 값 설정

```python
mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.2)
```

- 단순 코사인 유사도만 사용한 것과 비슷한 결과
- 높은 diversity 값 설정

```python
mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7)
```

- 다양한 키워드를 생성한다.