# 정렬-1

- 데이터를 큰 순서 혹은 작은 순서로 정렬하는 것



## 1. 선택 정렬

- 처리되지 않은 데이터 중에서 가장 작은 데이터를 선택해 맨 앞에 있는 데이터와 바꾸는 것을 반복
- 가장 기본적인 정렬



## 1) 선택 정렬 코드(파이썬)

```python
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(len(array)):
    min_index = i # 가장 작은 원소의 인덱스
    for j in range(i + 1, len(array)):
        if array[min_index] > array[j]:
            min_index = j
    array[i], array[min_index] = array[min_index], array[i] # 스왑

print(array)
```

<실행결과>

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

- 시간 복잡도 : O(N**2)



## 2. 삽입 정렬

- 처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입한다.
- 선택 정렬에 비해 구현 난이도가 높은 편이지만, 일반적으로 더 효율적으로 동작한다.



## 1) 삽입 정렬 코드(파이썬)

```python
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(1, len(array)):
    for j in range(i, 0, -1): # 인덱스 i부터 1까지 1씩 감소하며 반복하는 문법
        if array[j] < array[j - 1]: # 한 칸씩 왼쪽으로 이동
            array[j], array[j - 1] = array[j - 1], array[j]
        else: # 자기보다 작은 데이터를 만나면 그 위치에서 멈춤
            break

print(array)
```

<실행결과>

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

- 시간 복잡도 : O(N**2)
  - 최선의 경우 O(N) : 현재 리스트의 데이터가 거의 정렬되어 있는 상태라면 매우 빠르게 동작한다.
  - 이미 정렬되어 있는 상태에서 삽입 정렬 수행하면 N -1 번 반복