# 한국어 데이터 전처리 예제 : 데이터 로드 ~ 정수인코딩

1. Data Load

   ```python
   from konlpy.corpus import kolaw
   t_data = kolaw.open('constitution.txt')
   ```

2. 문장 토큰화

   ```python
   from soynlp import DoublespaceLineCorpus
   t_data = DoublespaceLineCorpus(t_data.name, iter_sent=True)
   ```

3. 형태소 토큰화 : 길이가 1보다 큰 단어만 저장

   ```python
   from konlpy.tag import Okt
   tw = Okt()
   d = []
   for i in t_data:
       s_data = tw.morphs(i)
       s_data = [i for i in s_data if len(i) > 1]
       d.append(s_data)
   ```

4. 단어 토큰화 및 정수 인코딩

   ```python
   from tensorflow.keras.preprocessing.text import Tokenizer
   
   text_encoding = Tokenizer(filters='[-=+,#/\?:^$.@*\"※~&%ㆍ!』①②③④⑤⑥\\‘|\(\)\[\]\<\>`\'…》]')
   text_encoding.fit_on_texts(d)
   encoded_text = text_encoding.texts_to_sequences(d)
   ```

5. 패딩

   ```python
   from tensorflow.keras.preprocessing.sequence import pad_sequences
   
   end_data = pad_sequences(encoded_text, padding='post', truncating='post')
   end_data
   ```

6. 실행결과

   ```
   array([[ 70,  12,   0, ...,   0,   0,   0],
          [531, 532, 533, ...,   7, 141,   1],
          [  0,   0,   0, ...,   0,   0,   0],
          ...,
          [ 12,  42,  40, ...,   0,   0,   0],
          [373,  12,  84, ...,   0,   0,   0],
          [375,  12,  84, ...,   0,   0,   0]])
   ```