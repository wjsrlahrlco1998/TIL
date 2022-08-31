# BERT의 문장 임베딩을 이용한 한국어 챗봇

- BERT의 Sentence Embedding을 이용한 간단한 한국어 챗봇



## 1. 코드



### 1) 패키지 로드

```python
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import sentence_transformers
```

- 사용할 BERT는 sentence_transformers를 통해 사전학습된 모델을 사용한다.
- ``pip install sentence_transformers``를 통해 설치한다.



### 2) 한국어 Q&A 데이터 셋 로드

```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')

print(train_data.shape)
train_data.head()
```

![image-20220831164944919](C:\Users\Park Ji Seong\AppData\Roaming\Typora\typora-user-images\image-20220831164944919.png)



### 3) Pre-Trained BERT 모델 로드

```python
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
```

- ``xlm-r-100langs-bert-base-nli-stsb-mean-tokens`` 모델은 한국어가 포함된 다국어 모델이다.
- 100가지 언어를 지원(한국어 포함)하는 다국어 BERT BASE 모델로 SNLI 데이터를 학습 후 STS-B 데이터로 학습되었다.
- means tokens의 의미는 문장 표현을 얻기 위해서 평균 풀링(mean-tokens)를 사용했다는 의미이다.
- NLI 데이터 학습 후, STS 데이터로 추가 파인 튜닝한 모델이다.

- 더 다양한 모델은 https://huggingface.co/models?library=sentence-transformers 에서 확인 가능하다.



### 4) 임베딩

```python
%%time
train_data['embedding'] = train_data.apply(lambda row: model.encode(row.Q), axis = 1)
```

- 데이터에서 모든 질문열, 즉, train_data['Q']에 대해서 문장 임베딩 값을 구한 후 embedding이라는 새로운 열에 저장한다.

![image-20220831165524978](C:\Users\Park Ji Seong\AppData\Roaming\Typora\typora-user-images\image-20220831165524978.png)



### 5) 테스트

```python
# 두 벡터간 코사인 유사도를 구하는 함수
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))
```

```python
def return_answer(question):
    embedding = model.encode(question)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return train_data.loc[train_data['score'].idxmax()]['A']
```

- return_answer 함수는 임의의 질문이 들어오면 해당 질문의 문장 임베딩 값과 챗봇 데이터의 임베딩 열. 즉, train_data['embedding']에 저장해둔 모든 질문 샘플들의 문장 임베딩 값들을 전부 비교하여 코사인 유사도 값이 가장 높은 질문 샘플을 찾아낸다. 
- 해당 질문 샘플과 짝이 되는 답변 샘플을 리턴한다.

![image-20220831165538525](C:\Users\Park Ji Seong\AppData\Roaming\Typora\typora-user-images\image-20220831165538525.png)

