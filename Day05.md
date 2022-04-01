# HTML 데이터 분석기법

## 1. requests 모듈



#### 1) requests 모듈을 이용한 크롤링

- 장점

  - 간단하게 크롤링이 가능하다

- 사용예시

  ```python
  import requests
  r=requests.get("URL") # URL로부터 데이터를 받는다.
  r.raise_for_status() # 접속하지 못한 경우 예외 발생
  print(r.status_code) # URL의 상태
  print(r.text) # URL의 내용을 txt 형태로 출력
  ```

#### 2) 정규 표현식을 이용한 분석

- 사용 모듈

  ``import re``

- 키워드 3가지

  1. ``.{문자}``

     ex) .ata : ?ata로 시작하는 문자열를 찾음 (?에는 어떠한 문자가 들어가도 된다)

  2. ``^{문자}``

     ex) ^ata : ata로 시작하는 문자열 찾음

  3. ``{문자}$``

     ex) ata$ : ata로 끝나는 문자열을 찾음

- 주요 기능
  1. ``match("문자열"): 처음부터 일치``
  2. ``search("문자열"): 일치하는 문자 있는지 확인``
  3. ``findall("문자열"): 일치 하는 모든것의 리스트 출력``

#### 3) 예제

```python
# 임의 리스트 선언
l = ['abcd','adcd','accd','abdc']
# compile('찾을 문자열')
ck=re.compile("a.cd")

# 일치하는 문자가 있을 경우 양식에 맞게 출력
def print_t(str):
    if str:
        print("일치문자",str.group())
        print("입력문자", str.string)
        print("일치문자 시작", str.start())
        print("일치문자 끝", str.end())
        print("일치문자 시작,끝", str.span())
    else:
        print("일치 없음")

for i in l:
    str =ck.match(i)
    print_t(str)
    str=ck.search(i)
    print_t(str)
    print("all_data",ck.findall(i))
```