# 스택과 큐



## 1. 스택



- 선입후출 : 먼저 입력되는 데이터가 나중에 출력

- 출구가 하나인 통에 데이터를 넣는다고 생각



### 1) 파이썬 코드

```python
stack = []

stack.append(5) # 원소 삽입
stack.append(3) # 원소 삽입
stack.pop() # 원소 삭제

print(stack[::-1]) # 최상단 원소부터 출력
print(stack) # 최하단 원소부터 출력
```

```
5
5
```



- append와 pop 연산은 시간복잡도가 1이므로 사용에 적합하다.



## 2. 큐



- 선입선출 : 먼저 들어온 데이터가 먼저 나감
- 입구와 출구가 모두 뚫여 있는 터널에 데이터를 넣는다고 생각
- 대기열과 같음.



### 1) 코드

```python
from collections import deque

# 큐 구현을 위한 deque 사용
queue = deque()

queue.append(5) # 삽입
queue.append(2) # 삽입
queue.append(3) # 삽입
queue.popleft() # 삭제

print(qeueu) # 먼저 들어온 순서대로 출력
queue.reverse() # 역순으로 바꾸기
print(queue) # 나중에 들어온 원소부터 출력
```

```
[3, 2]
[2, 3]
```

- deque가 빠르기 때문에 deque로 구현하여 사용.
