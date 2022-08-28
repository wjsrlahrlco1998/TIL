# BiLSTM을 이용한 한국어 스팀 리뷰 감성 분류

> 긍정 또는 부정으로 이진분류를 진행한다.



## 1. BiLSTM을 텍스트 분류에 사용하기

- 양방향 LSTM은 두 개의 독립적인 LSTM 아키텍쳐를 함께 사용하는 구조이다.
- Keras에서는 양방향 LSTM을 사용하면서 return_sequences=False로 택하면, 순방향 LSTM은 마지막 시점의 은닉 상태를 반환하고, 역방향 LSTM의 경우에는 첫번째 시점의 은닉 상태를 반환한다. 위 구조를 통해서 양방향 LSTM으로 텍스트 분류를 수행한다.



## 2. 코드



### 1) 패키지 로드

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 워닝 무시
import warnings
warnings.filterwarnings('ignore')
```



### 2) 데이터 로드 및 전처리

```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/steam.txt", filename="steam.txt")
```

- 해당 데이터에는 열 제목이 별도로 존재하지 않기 때문에, 임의로 두 개의 열 제목인 'label'과 'reviews'를 추가한다.

```python
total_data = pd.read_table('steam.txt', names=['label', 'reviews'])

print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력
total_data.head()
```

- 각 열에 대해서 중복을 제외한 샘플의 수를 카운트한다.

```python
total_data['reviews'].nunique()
total_data['label'].nunique()
```

> reviews열에서 중복을 제외한 경우 99,892개 이다.
> 현재 10만개의 리뷰가 존재하므로 이는 현재 갖고 있는 데이터에 중복인 샘플들이 있다는 의미이다.
> 중복인 샘플들을 제거해준다.

```python
total_data.drop_duplicates(subset=['reviews'], inplace=True) # reviews 열에서 중복인 내용이 있다면 중복 제거
print('총 샘플의 수 :',len(total_data))
```

- NULL 값의 유무 확인

```python
print(total_data.isnull().values.any())
```



### 3) 훈련 데이터와 테스트 데이터 분리

- 훈련 데이터 : 테스트 데이터 = 3 : 1의 비율로 분리

```python
train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42)
print('훈련용 리뷰의 개수 :', len(train_data))
print('테스트용 리뷰의 개수 :', len(test_data))
```

>훈련용 리뷰의 경우 약 7만 5,000개. 테스트용 리뷰의 경우 약 2만 5,000개가 존재한다.



### 4) 레이블의 분포 확인

- 데이터의 불균형 여부를 확인한다.

```python
train_data['label'].value_counts().plot(kind = 'bar')
print(train_data.groupby('label').size().reset_index(name = 'count'))
```

> 두 레이블 모두 약 3만 7천개로 50:50 비율을 가지고 있다.



### 5) 데이터 정제하기

- 정규 표현식을 사용하여 한글을 제외하고 모두 제거해준다. 
  또한 혹시 이 과정에서 빈 샘플이 생기지는 않는지 확인한다.

```python
# 한글과 공백을 제외하고 모두 제거
train_data['reviews'] = train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data['reviews'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())
```

- 테스트 데이터에 대해서도 같은 과정을 거친다.

```python
test_data.drop_duplicates(subset = ['reviews'], inplace=True) # 중복 제거
test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['reviews'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))
```

- 불용어 정의

```python
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']
```



### 6) 토큰화

```python
from konlpy.tag import Mecab
mecab = Mecab('C:\mecab\mecab-ko-dic')

train_data['tokenized'] = train_data['reviews'].apply(mecab.morphs)
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
test_data['tokenized'] = test_data['reviews'].apply(mecab.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
```



### 7) 단어와 길이 분포 확인

- 긍정 리뷰에는 주로 어떤 단어들이 많이 등장하고, 부정 리뷰에는 주로 어떤 단어들이 등장하는지 두 가지 경우에 대해서 각 단어의 빈도수를 계산한다. 각 레이블에 따라서 별도로 단어들의 리스트를 저장해준다.

```python
negative_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)
```

- Counter()를 사용하여 각 단어에 대한 빈도수를 카운트한다. 
  우선 부정 리뷰에 대해서 빈도수가 높은 상위 20개 단어들을 출력한다.

```python
negative_word_count = Counter(negative_words)
print(negative_word_count.most_common(20))
```

- 긍정 리뷰에 대해서도 동일하게 출력한다.

```python
positive_word_count = Counter(positive_words)
print(positive_word_count.most_common(20))
```

- 두 가지 경우에 대해 각각 길이 분포를 확인한다.

```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
text_len = train_data[train_data['label']==1]['tokenized'].map(lambda x: len(x))
ax1.hist(text_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('긍정 리뷰의 평균 길이 :', np.mean(text_len))

text_len = train_data[train_data['label']==0]['tokenized'].map(lambda x: len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Negative Reviews')
fig.suptitle('Words in texts')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
print('부정 리뷰의 평균 길이 :', np.mean(text_len))
plt.show()
```

> 유의미한 차이는 존재하지 않는다.

```python
X_train = train_data['tokenized'].values
y_train = train_data['label'].values
X_test= test_data['tokenized'].values
y_test = test_data['label'].values
```



### 8) 정수 인코딩

- 기계가 텍스트를 숫자로 처리할 수 있도록 훈련 데이터와 테스트 데이터에 정수 인코딩을 수행해야 한다.
  우선, 훈련 데이터에 대해서 단어 집합(vocaburary)을 만든다.

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
```

> 단어 집합이 생성되는 동시에 각 단어에 고유한 정수가 부여된다.
> 이는 tokenizer.word_index를 출력하여 확인 가능하다.

```python
tokenizer.word_index
```

- 등장 횟수가 1회인 단어들은 자연어 처리에서 배제하고자 한다.
  이 단어들이 이 데이터에서 얼만큼의 비중을 차지하는지 확인해보자.

```python
threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```

> 단어가 약 32,000개가 존재합니다. 등장 빈도가 threshold 값인 2회 미만. 즉, 1회인 단어들은 단어 집합에서 약 42%를 차지합니다. 하지만, 실제로 훈련 데이터에서 등장 빈도로 차지하는 비중은 매우 적은 수치인 약 1.2%밖에 되지 않습니다. 
> 아무래도 등장 빈도가 1회인 단어들은 자연어 처리에서 별로 중요하지 않을 듯 합니다. 그래서 이 단어들은 정수 인코딩 과정에서 배제시키겠습니다. 등장 빈도수가 1인 단어들의 수를 제외한 단어의 개수를 단어 집합의 최대 크기로 제한하겠습니다.

```python
# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :',vocab_size)
```

> 단어 집합의 크기는 18,941개입니다. 
> 이를 토크나이저의 인자로 넘겨주면, 토크나이저는 텍스트 시퀀스를 숫자 시퀀스로 변환합니다. 
> 이러한 정수 인코딩 과정에서 이보다 큰 숫자가 부여된 단어들은 OOV로 변환하겠습니다.

```python
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
```

- 정수 인코딩이 진행되었는지 확인하고자 X_train과 X_test에 대해서 상위 3개의 샘플만 출력합니다.

```python
print(X_train[:3])
print(X_test[:3])
```



### 9) 패딩

- 서로 다른 길이의 샘플들의 길이를 동일하게 맞춰주는 패딩 작업을 진행해보겠습니다. 
  전체 데이터에서 가장 길이가 긴 리뷰와 전체 데이터의 길이 분포를 알아보겠습니다.

```python
print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```

> 리뷰의 최대 길이는 64, 평균 길이는 약 15입니다. 
> 그래프로 봤을 때, 전체적으로는 60이하의 길이를 가지는 것으로 보입니다.

```python
def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))
```

- 최대 길이가 64이므로 만약 60으로 패딩할 경우, 몇 개의 샘플들을 온전히 보전할 수 있는지 확인합니다.

```python
max_len = 60
below_threshold_len(max_len, X_train)
```

- 훈련용 리뷰의 99.99%가 60이하의 길이를 가집니다. 
  훈련용 리뷰를 길이 60으로 패딩하겠습니다.

```python
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
```



### 10) BiLSTM으로 스팀 리뷰 감성 분석

```python
import re
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(hidden_units))) # Bidirectional LSTM을 사용
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=256, validation_split=0.2)
```

```python
loaded_model = load_model('best_model.h5')
print("테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
```



### 11) 리뷰 예측

```python
def sentiment_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = mecab.morphs(new_sentence) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))
```

```python
sentiment_predict('노잼 ..완전 재미 없음 ㅉㅉ')
```

```python
sentiment_predict('조금 어렵지만 재밌음ㅋㅋ')
```

```python
sentiment_predict('케릭터가 예뻐서 좋아요')
```



