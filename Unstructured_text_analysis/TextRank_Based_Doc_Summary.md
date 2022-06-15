# 문장 임베딩 기반 텍스트 랭크(TextRank Based on Sentence Embedding)



## 1. 텍스트 랭크(TextRank)란?

- 텍스트 랭크 알고리즘의 기반은 페이지 랭크 알고리즘이다.
  - 페이지 랭크 알고리즘은 웹 페이지의 순위를 정하기 위해 사용되던 알고리즘이다.



## 2. Pre-Trained Glove 다운로드

### 1) 패키지 로드

```
import numpy as np
import gensim
from urllib.request import urlretrieve, urlopen
import gzip
import zipfile
```



### 2) Glove 다운로드

```python
urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")
zf = zipfile.ZipFile('glove.6B.zip')
zf.extractall() 
zf.close()

# glove 사전
glove_dict = dict()
f = open('glove.6B.100d.txt', encoding="utf8") # 100차원의 GloVe 벡터를 사용

for line in f:
    word_vector = line.split()
    word = word_vector[0]
    word_vector_arr = np.asarray(word_vector[1:], dtype='float32') # 100개의 값을 가지는 array로 변환
    glove_dict[word] = word_vector_arr
f.close()
```



## 3. 텍스트 랭크를 이용한 텍스트 요약



### 1) 패키지 로드

```python
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from urllib.request import urlretrieve
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
```



### 2) 데이터 로드

- NLTK에서 제공하는 불용어 다운로드

```python
stop_words = stopwords.words('english')
```

- 테니스 관련 기사 다운로드, 데이터 프레임 저장 => 텍스트 요약에 사용할 문장들

```python
urlretrieve("https://raw.githubusercontent.com/prateekjoshi565/textrank_text_summarization/master/tennis_articles_v4.csv", filename="tennis_articles_v4.csv")
data = pd.read_csv("tennis_articles_v4.csv")
data.head()
```

![image-20220615114345200](TextRank_Based_Doc_Summary.assets/image-20220615114345200.png)



### 3) 전처리

- 문장 토큰화

```python
data = data[['article_text']]
data['sentences'] = data['article_text'].apply(sent_tokenize)
data
```

![image-20220615114454362](TextRank_Based_Doc_Summary.assets/image-20220615114454362.png)



- 토큰화 및 전처리 함수 정의

```python
# 토큰화 함수
def tokenization(sentences):
    return [word_tokenize(sentence) for sentence in sentences]

# 전처리 함수
def preprocess_sentence(sentence):
  # 영어를 제외한 숫자, 특수 문자 등은 전부 제거. 모든 알파벳은 소문자화
  sentence = [re.sub(r'[^a-zA-z\s]', '', word).lower() for word in sentence]
  # 불용어가 아니면서 단어가 실제로 존재해야 한다.
  return [word for word in sentence if word not in stop_words and word]

# 위 전처리 함수를 모든 문장에 대해서 수행. 이 함수를 호출하면 모든 행에 대해서 수행.
def preprocess_sentences(sentences):
    return [preprocess_sentence(sentence) for sentence in sentences]
```



- 전처리 진행

```python
data['tokenized_sentences'] = data['sentences'].apply(tokenization)
data['tokenized_sentences'] = data['tokenized_sentences'].apply(preprocess_sentences)
data
```

![image-20220615114534612](TextRank_Based_Doc_Summary.assets/image-20220615114534612.png)



### 4) Glove를 사용하여 문서 요약

- 사용할 Glove가 100차원이기 때문에 100차원의 영벡터 정의

```python
embedding_dim = 100
zero_vector = np.zeros(embedding_dim)
```



- 단어 벡터의 평균을 구하는 함수 정의, 문장의 길이가 0이면 100차원의 영벡터 리턴

```python
# 단어 벡터의 평균으로부터 문장 벡터를 얻는다.
def calculate_sentence_vector(sentence):
  if len(sentence) != 0:
    return sum([glove_dict.get(word, zero_vector) 
                  for word in sentence])/len(sentence)
  else:
    return zero_vector
```



- 각 문장에 대해서 벡터를 반환하는 함수 정의

```python
# 각 문장에 대해서 문장 벡터를 반환
def sentences_to_vectors(sentences):
    return [calculate_sentence_vector(sentence) 
              for sentence in sentences]
```



- 모든 문장에 대하여 문장 벡터 생성

```python
data['SentenceEmbedding'] = data['tokenized_sentences'].apply(sentences_to_vectors)
data[['SentenceEmbedding']]
```

![image-20220615114748722](TextRank_Based_Doc_Summary.assets/image-20220615114748722.png)



- 문장 벡터들 간의 코사인 유사도를 구하는 유사도 행렬 함수를 정의

```python
def similarity_matrix(sentence_embedding):
  sim_mat = np.zeros([len(sentence_embedding), len(sentence_embedding)])
  for i in range(len(sentence_embedding)):
      for j in range(len(sentence_embedding)):
        sim_mat[i][j] = cosine_similarity(sentence_embedding[i].reshape(1, embedding_dim),
                                          sentence_embedding[j].reshape(1, embedding_dim))[0,0]
  return sim_mat
```



- 유사도 행렬 구하기

```python
data['SimMatrix'] = data['SentenceEmbedding'].apply(similarity_matrix)
data['SimMatrix']
```

