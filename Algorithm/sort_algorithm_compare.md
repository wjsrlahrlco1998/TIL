# 정렬 알고리즘 비교



## 1. 정렬 종류 및 시간, 공간 복잡도

*단, 대부분의 경우에 표준 정렬 라이브러리는 최악의 경우에도 O(NlogN)의 시간 복잡도를 보장한다.



- 선택 정렬
  - 평균 시간 복잡도 : O(N**2)
  - 공간 복잡도 : O(N)
- 삽입 정렬
  - 평균 시간 복잡도 : O(N**2)
  - 공간 복잡도 : O(N)
- 퀵 정렬
  - 평균 시간 복잡도 : O(NlogN) 
  - 공간 복잡도 : O(N)
  - 최악 시간 복잡도 : O(N**2)
- 계수 정렬
  - 평균 시간 복잡도 : O(N + K)
  - 공간 복잡도 : O(N + K)
  - 데이터의 크기가 한정되어 있는 경우에 적합



### 1) 선택 정렬과 기본 정렬 라이브러리 수행 시간 비교 코드(파이썬)

```python
from random import randint
import time

# 배열에ㅐ 10,000개의 정수를 삽입
array = []
for _ in range(10000):
    # 1부터 100 사이의 랜덤한 정수
    array.append(randint(1, 100))

# 선택 정렬 프로그램 성능 측정
start_time = time.time()

# 선택 정렬 프로그램 소스코드
for i in range(len(array)):
    min_index = i # 가장 작은 인덱스
    for j in range(i + 1, len(array)):
        if array[min_index] > array[j]:
            min_index = j
    array[i], array[min_index] = array[min_index], array[i]

# 측정 종료
end_time = time.time()
# 수행 시간 출력
print('선택 정렬 성능 측정:', end_time - start_time)

# 배열을 다시 무작위 데이터로 초기화
array = []
for _ in range(10000):
    # 1부터 100 사이의 랜덤한 정수
    array.append(randint(1, 100))

# 기본 정렬 라이브러리 성능 측정
start_time = time.time()

# 기본 정렬 라이브러리 사용
array.sort()

# 측정 종료
end_time = time.time()
# 수행 시간 출력
print('기본 정렬 라이브러리 성능 측정:', end_time - start_time)
```

선택 정렬 성능 측정: 6.890566349029541
기본 정렬 라이브러리 성능 측정: 0.000997304916381836



## 2. 정렬 알고리즘 문제



### Q1. 두 배열의 원소 교체

```
<문제 설명>
- 동빈이는 두 개의 배열 A와 B를 가지고 있습니다. 두 배열을 N개의 원소로 구성되어 있으며, 배열의 원소는 모두 자연수 입니다.
- 동빈이는 최대 K 번의 바꿔치기 연산을 수행할 수 있는데, 바꿔치기 연산이란 배열 A에 있는 원소 하나와 배열 B에 있는 원소 하나를 골라서 두 원소를 서로 바꾸는 것을 말한다.
- 동빈이의 최종 목표는 배열 A의 모든 원소의 합이 최대가 되도록 하는 것이며, 여러분은 동빈이를 도와야 한다.
- N, K, 그리고 배열 A와 B의 정보가 주어졌을 때, 최대 K 번의 바꿔치기 연산을 수행하여 만들 수 있는 배열 A의 모든 원소의 합의 최댓값을 출력하는 프로그램을 작성하시오.

즉, 정해진 K번 만큼 B와 A의 원소를 바꾸어 A의 배열 합이 최댓 값이 되도록 하라.
```

- 해결방법
  - 배열 A에 대하여 오름차순 정렬, 배열 B에 대하여 내림차순 정렬

```python
n, k = map(int, input().split()) # N과 K를 입력 받기
a = list(map(int, input().split())) # 배열 A의 모든 원소를 입력 받기
b = list(map(int, input().split())) # 배열 B의 모든 원소를 입력 받기

a.sort() # 배열 A는 오름차순 정렬 수행
b.sort(reverse=True) # 배열 B는 내림차순 정렬 수행

# 첫 번째 인덱스부터 확인하며, 두 배열의 원소를 최대 K번 비교
for i in range(k):
    # A의 원소가 B의 원소보다 작은 경우
    if a[i] < b[i]:
        # 두 원소를 교체
        a[i], b[i] = b[i], a[i]
    else: # A의 원소가 B의 원소보다 크거나 같을 때, 반복문을 탈출
        break

print(sum(a))
```

5 3
1 2 5 4 3
5 5 6 6 5
26





