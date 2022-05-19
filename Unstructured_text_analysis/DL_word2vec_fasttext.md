# word2vec, FastText

- Word2Vec
  - 단어들을 독립적으로 생각하는것이 아닌, 연관성을 고려하여 임베딩하여 학습한다.
  - 단, 입력된 데이터  내에서만 한정된다는 단점이 있다. -> 새로운 단어에 대해서는 알지 못한다.
- FastText
  - word2vec가 새로운 단어에 약한 것을 개선한 임베딩 알고리즘
  - 노이즈에 강하다. 새로운 단어에 대해서 형태적 유사성을 고려한 벡터 값을 얻는다.
  - 즉, 유의어를 판별할 수 있다.



---

### 1. Word2Vec 및 FastText 설치

- ``pip install gensim``



### 2. 피터팬 내용을 통한 word2vec 테스트

1. 패키지 로드

   ```python
   from nltk.tokenize import sent_tokenize , word_tokenize
   import gensim
   from gensim.models import Word2Vec ,FastText
   from gensim.test.utils import common_texts
   ```

2. 데이터 로드 및 전처리

   ```python
   s_data=open('data1.txt','r',encoding='utf-8').read()
   s_data=s_data.replace('\n',' ')
   
   d_l=[]
   
   for 문장 in sent_tokenize(s_data):
       s_l=[]
       for 단어 in word_tokenize(문장):
           s_l.append(단어.lower())
       d_l.append(s_l)
   ```

3. Word2Vec를 통한 임베딩

   ```python
   w_m1=Word2Vec(d_l,min_count=1,vector_size=100,window=5)
   ```

   - Word2Vec(입력data,단어빈도수,임배딩된차원,window=단어갯수,sg=0,CBOW,1=skip_gram))

4. 단어비교(연관성 비교)

   ```python
   w_m1.wv.similarity('peter','wendy'), w_m1.wv.similarity('peter','hook')
   ```

   ```
   0.074393846, 0.027709894
   ```

5. 하이퍼파라미터 조정

   ```python
   w_m2=Word2Vec(d_l,min_count=1,vector_size=100,window=5,sg=1)
   
   w_m2.wv.sim('peter','wendy'),w_m2.wv.similarity('peter','hook')
   ```

   ```
   (0.40088683, 0.52016735)
   ```



### 3. FastText 테스트

1. 임베딩

   ```python
   m2=FastText('data1.txt',vector_size=4,window=3,min_count=1)
   ```

2. 단어 연관성 비교

   ```python
   m2.wv.similarity('peter','wendy'), m2.wv.similarity('peter','hook')
   ```

   ```python
   0.45924556, 0.043825187
   ```