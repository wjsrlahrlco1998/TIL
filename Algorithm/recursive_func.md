# 재귀 함수 (recursive function)



## 1. 재귀 함수란?

- 자기 자신을 다시 호출하는 함수

- 단순한 형태의 재귀 함수 예제

  - '재귀 함수를 호출합니다'라는 메세지를 무한히 출력

  - 어느 정도 출력하다가 최대 재귀 깊이 초과 메세지가 출력된다. (파이썬)

    ```python
    def recursive_function():
        print('재귀 함수를 호출합니다.')
        recursive_function()
    recursive_function()
    ```

  - 재귀 깊이를 완화하려면 관련 옵션을 설정하거나 stack 구조로 정의해야한다.

- 재귀 함수의 종료 조건은 반드시 명시해야 한다.

  - 종료 조건을 재대로 명시하지 않으면 함수가 무한히 호출될 수 있다.

  - 종료 조건을 포함한 재귀 함수 예제

    ```python
    def recursive_function(i):
        # 100번째 호출을 했을 때 종료되도록 종료 조건 명시
        if i == 100:
            return
        print(i, '번째 재귀함수에서', i + 1, '번째 재귀함수를 호출합니다.')
        recursive_function(i + 1)
        print(i, '번째 재귀함수를 종료합니다.')
    
    recursive_function(1)
    ```

    

## 2. 팩토리얼 구현 예제 (파이썬)

```python
# 반복적으로 구현한 n!
def factorial_iterative(n):        
    result = 1
    # 1부터 n까지의 수를 차례대로 곱하기
    for i in range(1, n + 1):
       result *= i
    return result

# 재귀적으로 구현한 n!
def factorial_recursive(n):        
    if n <= 1: # n이 1 이하인 경우 1을 반환
        return 1
    # n! = n * (n - 1)!를 그대로 코드로 작성하기
    return n * factorial_recursive(n - 1)

# 각각의 방식으로 구현한 n! 출력(n = 5)
print('반복적으로 구현:', factorial_iterative(5))
print('재귀적으로 구현:', factorial_recursive(5))
```



## 3. 최대공약수 계산 (유클리드 호제법) 예제

- 두 개의 자연수에 대한 최대공약수를 구하는 대표적인 알고리즘은 유클리드 호제법이다.

- 유클리드 호제법

  - 두 자연수 A, B에 대하여 (A > B) A를 B로 나눈 나머지를 R이라고 한다.
  - 이때 A와 B의 최대공약수는 B와 R의 최대공약수와 같다.

- 유클리드 호제법의 아이디어를 그래도 재귀함수로 작성할 수 있다.

  ```python
  def gcd(a, b):
      if a % b == 0:
          return b
      else:
          return gcd(b, a % b)
  
  print(gcd(192, 162))    
  ```

  < 실행결과>

  6



## 4. 재귀함수 유의사항

- 재귀 함수를 잘 활용하면 복잡한 알고리즘을 간결하게 작성할 수 있다.
  - 하지만 가독성이 떨어질 수 있다.
- 모든 재귀함수는 반복문을 이용하여 동일한 기능을 구현할 수 있다.
- 재귀함수가 반복문보다 유리한 경우도 있고 불리한 경우도 있다.
- 컴퓨터가 함수를 연속적으로 호출하면 컴퓨터 메모리 내부의 스택 프레임에 쌓인다.
  - 그래서 스택을 사용할 때 구현상 스택 라이브러리 대신에 재귀 함수를 이용하는 경우가 많다.