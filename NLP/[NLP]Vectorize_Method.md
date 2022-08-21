# 자연어에서 벡터가 의미를 담는 방법



## 1. 임베딩을 만드는 3가지 방법



### 1) Bag of Words

- 단어의 순서는 고려하지 않고 **단어의 빈도 수**와 **등장 여부**만을 고려하여 임베딩하는 기법

- 많이 쓰인 단어가 주제와 더 강한 관련을 맺는다는 전제
- **정보 검색**에 아직도 많이 활용되는 방법
- 통계적 기법으로 **TF-IDF** 방식을 사용한다.



#### (1) TF-IDF(Term Frequency-Inverse Document Frequency)

- TF : 어떤 단어가 특정 문서에 많이 쓰였는지 빈도를 나타낸다.
- DF : 특정 단어가 나타난 문서의 수를 나타낸다.

(TF는 같은 단어라도 문서마다 다른 값을 가지고, DF는 문서가 달라지더라도 단어가 같다면 동일한 값을 가진다)

- IDF : 전체 문서의 수(N)을 해당 단어의 DF로 나눈 뒤 로그를 취한 값으로, **값이 클수록 특이한 단어**라는 의미이다. (이 값은 단어의 주제 예측 능력과 직결된다.) -> 어떤 단어의 주제 예측 능력이 강할 수록 가중치가 커지고 그 반대의 경우 작아진다.



#### (2) Deep Averaging Network

- Bag of Words 가정의 뉴럴 네트워크 버전
- Bag of Words와 마찬가지로 단어의 순서를 고려하지 않는다.



### 2-1) 통계 기반 언어 모델

- 언어 모델이란 단어 시퀀스에 확률을 부여하는 모델이다.

- 시퀀스 정보를 명시적으로 학습한다. 
- 자연스러운 문장에 높은 확률 값을 부여한다.

```
Ex)
누명을 쓰다 -> 0.41
누명을 당하다 -> 0.02

두시 삼십이분 -> 0.51
이시 서른두분 -> 0.08
```

- n-gram 을 이용한다.



#### (1) back-off

- n-gram 등장 빈도를 n보다 작은 범위의 단어 시퀀스 빈도로 근사하는 방식
  - n을 크게 할수록 등장하지 않는 케이스가 많아진다.

#### (2) smoothing

- 등장 빈도 표에 모두 k만큼 더하는 기법
  - k를 1로 설정하면 laplace smoothing이라고 한다.
- 이 방법을 사용하면 전혀 등장하지 않는 케이스들에는 작으나마 일부 확률을 부여하게 된다.



### 2-2) 뉴럴 네트워크 기반 언어모델

- 뉴럴 네트워크 기반의 언어모델은 입력과 출력 사이의 관계를 유연하게 포착해낼 수 있다.
  - 그 자체로 확률 모델로 기능할 수 있다.



#### (1) masked language model

- ex. 발 없는 말이 -> 언어모델 -> 천리 와 같이 해당 마스크 위치에 어떤 단어가 올지 예측하는 과정에서 학습한다.
- 문장 전체를 다 보고 중간에 있는 단어를 예측하기 때문에 양뱡향 학습이 가능하다. 따라서 기존 언어 모델보다 임베딩 품질이 좋다.
- 예시로 BERT가 이에 속한다.
