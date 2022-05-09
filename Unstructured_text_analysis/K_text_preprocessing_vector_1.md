# 한글 전처리 : 벡터화 기초



## 1. DictVectorizer

- BOW 인코딩 벡터



### 1) 간단한 사용 예시

- import

  ``from sklearn.feature_extraction import DictVectorizer``

- 벡터화

  ```python
  v = DictVectorizer(sparse=False)
  D = [{'A' : 1, 'B' : 2}, {'B' : 3, 'C' : 1}]
  X = v.fit_transform(D)
  print(X) # A, B, C의 3종의 단어를 하나의 단어로 묶음, 각 값은 각 문자별 빈도수, 열은 문자의 종류, 대괄호는 문장을 의미
  ```

  실행결과 :

  ```
  [[1. 2. 0.]
   [0. 3. 1.]]
  ```

  - 각 열은 단어의 종류
  - 각 행은 문장 별 단어 별 개수

- 단어의 종류 확인

  ``v.feature_names_``

- 학습하지 않은 단어는 벡터화되지않음

  ```python
  D2 = {'A' : 5, 'B' : 1, 'D': 100}
  v.transform(D2) # D는 학습하지 않았기 때문에 표시되지 않는다.
  ```

  실행결과 :

  ```
  array([[5., 1., 0.]])
  ```



## 2. CountVectorizer

- BOW 인코딩 벡터 문서의 집합 -> 단어 토큰 생성 -> 각 단어의 수 확인



### 1) 간단한 사용예시

- import

  ``from sklearn.feature_extraction.text import CountVectorizer``

- 사용할 예시 문서 (문장 토큰화 완료)

  ```python
  corpus = [
      'This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?',
      'The last document?',
  ]
  ```

- 단어 토큰 생성 및 각 단어의 수 확인

  ```python
  # 단어 토큰 생성(단어 인덱스 생성)
  v1 = CountVectorizer()
  # 각 단어의 수
  v1.fit(corpus)
  v1.vocabulary_
  ```

  실행결과 :

  ```
  {'this': 9,
   'is': 3,
   'the': 7,
   'first': 2,
   'document': 1,
   'second': 6,
   'and': 0,
   'third': 8,
   'one': 5,
   'last': 4}
  ```

- 테스트

  ```python
  v1.transform(['This is the first document. This This']).toarray()
  v1.transform(['This is the first document. data']).toarray()
  ```

  실행결과 :

  ```
  array([[0, 1, 1, 1, 0, 0, 0, 1, 0, 3]], dtype=int64)
  array([[0, 1, 1, 1, 0, 0, 0, 1, 0, 1]], dtype=int64)
  ```

  - 입력으로는 문서의 집합(문장의 집합)으로 전달해야한다.
  - 원-핫 인코딩과는 다르다
  - 학습되지않은 단어는 벡터화 대상에 포함되지 않는다.