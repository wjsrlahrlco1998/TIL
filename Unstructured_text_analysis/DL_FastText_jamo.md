# 자모 단위 토큰화와 FastText 임베딩



## 1. 자모 단위 토큰화를 하는 이유

- FastText 임베딩 모델의 특성은 단어 토큰의 문자 하나하나를 고려하여 비슷한 단어를 찾을 수 있다는 것이다. 예를들면, 일반 Word2Vec는 '남동생'과 '동생'은 다른 단어로 취급한다. 하지만 FastText는 '남동', '동생'으로 분해하여 '동생'과 비슷하다는 것을 예측한다.
- FastText에 자음 모음 단위로 토큰화된 단어를 학습시키면 어떨까? 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'으로 토큰화된 단어를 학습시키면 '냠동생', '뎡생' 등의 단어가 입력으로 들어와도 남동생과 비슷하다는 것을 예측할 수 있다. 
- 즉, FastText가 노이즈에 강하다는 특징을 극대화 시킬 수 있다.



## 2. 자모 단위 토큰화 코드

```python
# 단어 -> 자모 변환 함수
def word_to_jamo(token):
        '''자모 토큰화 함수'''
        decomposed_token = ''
        for char in token:
            try:
                # char(음절)을 초성, 중성, 종성으로 분리
                cho, jung, jong = hgtk.letter.decompose(char)
                
                # 자모가 빈 문자일 경우 특수문자 -로 대체
                cho = to_special_token(cho)
                jung = to_special_token(jung)
                jong = to_special_token(jong)
                decomposed_token = decomposed_token + cho + jung + jong
                
            # 만약 char(음절)이 한글이 아닐 경우 자모를 나누지 않고 추가
            except Exception as exception:
                if type(exception).__name__ == 'NotHangulException':
                    decomposed_token += char
        
        return decomposed_token

def to_special_token(jamo):
        '''자모 검사 함수'''
        if not jamo:
            return '-'
        else:
            return jamo
```

이 함수는 단어를 자음 모음단위로 분해한다.



## 3. 자모 단위로 토큰화된 단어를 합치는 함수

```python
def jamo_to_word(jamo_sequence):
  tokenized_jamo = []
  index = 0
  
  # 1. 초기 입력
  # jamo_sequence = 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'

  while index < len(jamo_sequence):
    # 문자가 한글(정상적인 자모)이 아닐 경우
    if not hgtk.checker.is_hangul(jamo_sequence[index]):
      tokenized_jamo.append(jamo_sequence[index])
      index = index + 1

    # 문자가 정상적인 자모라면 초성, 중성, 종성을 하나의 토큰으로 간주.
    else:
      tokenized_jamo.append(jamo_sequence[index:index + 3])
      index = index + 3

  # 2. 자모 단위 토큰화 완료
  # tokenized_jamo : ['ㄴㅏㅁ', 'ㄷㅗㅇ', 'ㅅㅐㅇ']
  
  word = ''
  try:
    for jamo in tokenized_jamo:

      # 초성, 중성, 종성의 묶음으로 추정되는 경우
      if len(jamo) == 3:
        if jamo[2] == "-":
          # 종성이 존재하지 않는 경우
          word = word + hgtk.letter.compose(jamo[0], jamo[1])
        else:
          # 종성이 존재하는 경우
          word = word + hgtk.letter.compose(jamo[0], jamo[1], jamo[2])
      # 한글이 아닌 경우
      else:
        word = word + jamo

  # 복원 중(hgtk.letter.compose) 에러 발생 시 초기 입력 리턴.
  # 복원이 불가능한 경우 예시) 'ㄴ!ㅁㄷㅗㅇㅅㅐㅇ'
  except Exception as exception:  
    if type(exception).__name__ == 'NotHangulException':
      return jamo_sequence

  # 3. 단어로 복원 완료
  # word : '남동생'

  return word
```



