# LSTM(Long Short-Term Memory)



## 1. LSTM(Long Short-Term Memory) 이란?

- 바닐라 RNN(Vanilla RNN)의 장기 의존성 문제(the problem of Long-Term Dependencies)를 개선한 모델
  - 장기 의존성 문제(the problem of Long-Term Dependencies)는 RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이다.

![image-20220531172117089](DL_RNN_2.assets/image-20220531172117089.png)

- 전통적인 RNN의 단점을 보완한 이러한 RNN의 일종을 장단기 메모리(Long Short-Term Memory)라고 하며, 줄여서 LSTM이라고 한다.
- LSTM은 RNN과 비교하여 긴 시퀀스의 입력을 처리하는데 탁월한 성능을 보인다.

- 은닉 상태의 값과 셀 상태의 값을 구하기 위해서 새로 추가 된 3개의 게이트를 사용한다. 각 게이트는 삭제 게이트, 입력 게이트, 출력 게이트라고 부르며 이 3개의 게이트에는 공통적으로 시그모이드 함수가 존재한다.

