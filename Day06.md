# HTML 데이터 분석기법-2

## 2. BeautifulSoup 모듈

- 기능

  ``requests``로 가져온 데이터를 원하는 형태로 변환

  *soup는 데이터를 찾는 것에 특화되어있다.*

- 명령어

  ```python
  # url로부터 데이터 가져옴
  r = requests.get(url)
  
  # 1. lxml 형식 : lxml은 html과 흡사하나 데이터 관점에서 가져온다는 점에서 차이가 있다.
  soup = BeautifulSoup(r.text,"lxml")
  
  # 2. html.parser 형식 : lxml과 동일한 기능이나 형태에서 차이가 있다.
  soup = BeautifulSoup(r.text,"html.parser")
  ```

- 변환한 데이터에서 원하는 데이터 추출(예시)

  ``soup.title.get_text()`` : title이라는 클래스의 데이터를 text로 추출한다.

  ``soup.a['href']`` : a 태그의 href의 데이터를 가져온다.

  ``soup.find('a',attrs={"키":"값"})``: 찾으려는 태그의 클래스와 값을 키-값 형식으로 찾는다.

#### 1) 전체 예시 코드(데이터 추출)

```python
import requests
from bs4 import BeautifulSoup

# URL지정
url='URL'
# URL로부터 HTML 데이터 가져오기
r=requests.get(url)
# 가져온 HTML 데이터를 지정한 형태로 변환
soup=BeautifulSoup(r.text,"html.parser")

# 1. 하나의 원소만 가져오는 경우
# a 태그 데이터를 가까운 순서에서 찾아서 하나 가져온다.
tag_1 = soup.find('a')

# 2. 복수의 원소를 가져오는 경우
# a 태그 데이터를 모두 가져온다.
tag_2 = soup.find_all('a')
```



#### 2) 부가 기능

- 가져온 태그에서 다음 요소를 가리키는 법

  ```python
  # a태그의 하나의 원소를 가져옴
  data = soup.find('a')
  
  # data가 가리키던 원소의 다음 원소를 가리킴
  data_2 = data.next_sibling 
  ```

- 내가 원하는 정보의 내용을 선택해서 가져오는 법

  ``soup.select('td[class=title]')`` : td 태그의 class 값이 title인 데이터를 가져옴

  ``soup.select('td.title')`` :  td 태그의 class 값이 title인 데이터를 가져옴

