# 한글 비정형 텍스트 전처리 - 2

---

[TOC]

---

## 패키지 설치

1. 한글 띄워쓰기 교정 패키지
   - ``pip install git+https://github.com/haven-jeon/PyKoSpacing.git``
2. 한글 맞춤법 교정 패키지
   - ``pip install git+https://github.com/ssut/py-hanspell.git``
3. 한글 자연어 처리 기본 패키지
   - ``pip install soynlp``
4. 한글 형태소 보조 분석 패키지
   - ``pip install customized_konlpy``

---

## 1. 한글 전처리 : 띄워쓰기, 맞춤법 교정

### 1) 띄워쓰기 교정

### * import 

- ``from pykospacing import Spacing``
  - 자동 띄워쓰기 패키지
  - 형태소를 기반하여 띄워쓰기
  - 완벽하지는 않지만 정리목적으로 좋다.

### * 코드

1. 테스트할 텍스트

   ```python
   text = "그러면서정씨는9일국회인사청문회를앞두고있는한동훈법무부장관후보자자녀논란과관련해."
   ```

2. Spacing 사용

   ```python
   sc = Spacing()
   s_text = sc(text)
   s_text
   ```

   결과 : '그러면서 정씨는 9일 국회 인사청문회를 앞두고 있는 한동훈 법무부 장관 후보자 자녀 논란과 관련해.'

---

### 2) 맞춤법 교정

### * import

- ``from hanspell import spell_checker``
  - 맞춤법 교정 패키지
  - 오타 수정 및 띄워쓰기 교정

### * 코드

1. 테스트할 텍스트

   ```python
   text = '나는 외 않되? 나도 할수있으면 돼지'
   ```

2. spell_checker 사용

   ```python
   ck_t = spell_checker.check(text)
   ck_t.checked
   ```

   실행결과 : '나는 왜 안돼? 나도 할 수 있으면 되지'

---

## 2. 한글 전처리 : 자연어 처리기

### 1) 문장 토큰화 및 단어 학습

### * import

- ``from soynlp import DoublespaceLineCorpus``
  - 문장 토큰화를 위한 패키지
- ``from soynlp.word import WordExtractor``
  - 단어 학습을 위한 패키지

[DOC(상세설명)](https://github.com/lovit/soynlp)

### * 코드

- 데이터 로드 

  1. 문서 단위로 토큰화하여 load

     ```python
     all_data = DoublespaceLineCorpus('data1.txt')
     ```

  2. 문장 단위로 토큰화하여 load

     ```python
     all_data = DoublespaceLineCorpus('data1.txt', iter_sent=True)
     ```

- 데이터 학습

  ```python
  w_e = WordExtractor()
  w_e.train(all_data) # 학습(처리)
  w_e_t = w_e.extract() # 처리된 데이터 반환
  ```

- 단어의 응집 점수 확인

  ```python
  w_e_t['반포한강공원에'].cohesion_forward
  ```

  실행결과 : 0.33492963377557666

- 단어의 엔트로피 값 확인

  ```python
  w_e_t['반포한강공원'].right_branching_entropy
  ```

  실행결과 : 1.3542944153448395

---

### 2) 토큰화 

*앞서 WordExtractor로 학습된 모델로 토큰화 진행*

#### (1) LTokenizer

- import

  ``from soynlp.tokenizer import LTokenizer``

- {word : score} 형태 dict 생성

  ```python
  sc = {w : scr.cohesion_forward for w, scr in w_e_t.items()}
  ```

- 학습된 모델을 기반하여 형태소로 나누기

  ```python
  l_tk = LTokenizer(scores=sc)
  ck_t = '반포한강공원에'
  l_tk.tokenize(ck_t, flatten=False)
  ```

  실행결과 : [('반포한강공원', '에')]

  학습되었던 문장은 잘 나눔.

#### (2) MaxScoreTokenizer

- import

  ``from soynlp.tokenizer import MaxScoreTokenizer``

  문장의 연결성 점수로 판단하여 분리

- {word : score} 형태 dict 생성

  ```python
  sc = {w : scr.cohesion_forward for w, scr in w_e_t.items()}
  ```

- 학습된 모델을 기반하여 문장 나누기

  ```python
  ck_t = '자료의정보를구분하기위해문서작성을했다' 
  m_tk = MaxScoreTokenizer(scores=sc)
  m_tk.tokenize(ck_t, flatten=False)
  ```

  실행결과 : 

  ```
  [[('자료', 0, 2, 0.05542850920178591, 2),
    ('의', 2, 3, 0.0, 1),
    ('정보를', 3, 6, 0.13143652206154155, 3),
    ('구분하기', 6, 10, 0.06920871053860748, 4),
    ('위해', 10, 12, 0.36709069882702794, 2),
    ('문서작성을', 12, 17, 0.0, 5),
    ('했다', 17, 19, 0.6890166650423936, 2)]]
  ```

​		유사성의 점수에 따라 분류를 다르게 한다.

​		위의 문장은 학습되지 않은 문장이라 제대로 나누어지지 않는다.

---

### 3) 줄임말 정형화

- import

  ``from soynlp.normalizer import *``

  중첩되는 글자에 대한 정리

- 테스트할 텍스트

  ```python
  review_text_1 = '영화가 너무 웃겨'
  review_text_2 = '영화가 너무 웃겨 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'
  review_text_3 = '영화가 너무 웃겨 ㅋㅋ'
  review_text_4 = '영화가 너무 웃겨 ㅋㅋㅋ'
  t_e = '후후후후후훗'
  ```

- 줄임말 정형화

  ```python
  emoticon_normalize(review_text_1)
  emoticon_normalize(review_text_2, num_repeats=1) # num_repeats 기본 값 2
  emoticon_normalize(review_text_3, num_repeats=3)
  emoticon_normalize(review_text_4, num_repeats=2)
  emoticon_normalize(t_e)
  ```

  실행결과 :

  ```
  '영화가 너무 웃겨'
  '영화가 너무 웃겨 ㅋ'
  '영화가 너무 웃겨 ㅋㅋ'
  '영화가 너무 웃겨 ㅋㅋ'
  '후후훗'
  ```

---

## 3. 한글 전처리 : 명사등록

*한글에서 명사를 추출할 때, '은호'라는 명사가 '은'과 '호'로 분리가되는 현상을 명사등록을 통해 해결한다*

### * import

- ``from konlpy.tag import Okt``
  - 한글 형태소 분석기
- ``from ckonlpy.tag import Twitter``
  - 한글 형태소 분석기 보조장치



### * 코드

- 테스트할 텍스트

  ```python
  text = '은호가 교실로 들어갔다'
  ```

- 잘못 분리되는 상황

  ```python
  tw = Okt() # 형태소 분석기 객체생성
  
  print(tw.nouns(text)) # 1. 명사 추출
  print(tw.morphs(text)) # 2. 형태소 추출
  print(tw.pos(text)) # 3. 품사 추출
  ```

  실행결과 :

  ```
  ['은', '호가', '교실']
  ['은', '호가', '교실', '로', '들어갔다']
  [('은', 'Noun'), ('호가', 'Noun'), ('교실', 'Noun'), ('로', 'Josa'), ('들어갔다', 'Verb')]
  ```

- 명사 등록

  ```python
  tw2 = Twitter()
  tw2.add_dictionary('은호', 'Noun')
  ```

- 명사 등록 후 결과 확인

  ```python
  print(tw2.nouns(text))
  print(tw2.morphs(text))
  print(tw2.pos(text))
  ```

  실행결과 :

  ```
  ['은호', '교실']
  ['은호', '가', '교실', '로', '들어갔다']
  [('은호', 'Noun'), ('가', 'Josa'), ('교실', 'Noun'), ('로', 'Josa'), ('들어갔다', 'Verb')]
  ```

---

## @ kss 문장 토큰화

*kss 패키지를 활용한 문장 토큰화*

#### * import

- ``import kss ``



#### * 코드

- 데이터 셋 로드

  ```python
  from konlpy.corpus import kobill
  t_data = kobill.open('1809890.txt').read()
  ```

- 데이터 정제

  ```python
  s_data = t_data.replace('\n', '')
  ```

  kss는 데이터의 불순물이 제거되어야 토큰화가 진행된다.

- 문장 토큰화

  ```python
  kss.split_sentences(s_data)
  ```

  

