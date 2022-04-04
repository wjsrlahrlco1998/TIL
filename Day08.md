# 1. SQL을 이용한 데이터 저장 및 활용

- 기본적인 사용(저장)

  ```python
  #1. sqlite3 모듈을 import
  import sqlite3
  
  #2. DB 연결
  f = sqlite3.connect("Ex.db")
  
  #3. 커서지정 : 시작지점
  c = f.cursor()
  
  #4. 테이블 초기화
  c.execute('DROP TABLE IF EXISTS {테이블이름}')
  
  #5. 테이블 생성(변수 선언 : 변수명 자료형)
  c.execute('''CREATE TABLE {테이블이름}(
      {변수명} {변수자료형}, 
      ...
      {변수명} {변수자료형}
  )''')
  
  #6. 테이블의 변수 값 삽입
  #6-1. 한 줄 삽입 : 나열식 입력
  c.execute('INSERT INTO {테이블이름} VALUES(?,?,?)', (10,20,30))
  #6-2. 한 줄 삽입 : 키-값 입력
  c.execute('INSERT INTO {테이블이름} VALUES(:{키이름},:{키이름},:{키이름})', {'{키이름}':값,'{키이름}':값,'{키이름}':값})
  #6-3. 여러 줄 삽입 : 여러 줄 작성(제공된 data(list)의 길이로 반복횟수 결정, dict형을 가지는 리스트를 전달해야한다.
  c.executemany('INSERT INTO {테이블이름} VALUES(:{키이름},:{키이름},:{키이름})',
                {키가 일치하는 dict 자료형을 가진 list 형})
  
  #7. 작업 마무리(저장)
  f.commit()
  
  #8. 연결닫기
  f.close()
  ```

  

- 기본적인 사용(출력)

  ```python
  #1. sqlite3 모듈을 import
  import sqlite3
  
  #2. DB 연결
  f = sqlite3.connect("Ex.db")
  
  #3. 커서지정 : 시작지점
  c = f.cursor()
  
  #4. 데이터 선택 : {테이블이름}으로부터 모든 데이터를 가져옴
  c.execute('SELECT * FROM {테이블이름}')
  
  #5. 데이터 꺼냄 : [(키,값),...,(키,값)]의 형식으로 반환된다.
  for i in c.fetchall():
      print(f"키이름:{i[0]},키이름:{i[1]},키이름:{i[2]}")
  ```

  

- 예제 : 네이버 영화에서 리뷰 추출 후 SQL로 저장 및 출력

  ```python
  import requests
  from bs4 import BeautifulSoup
  import sqlite3
  
  review_dict_list = []
  review_dict = dict()
  url = "https://movie.naver.com/movie/point/af/list.naver?&page=10"
  
  def 수집():
      global review_dict_list
      global review_dict
      global url
  
      r = requests.get(url)
      html_d = r.text
      soup = BeautifulSoup(html_d, "html.parser")
      data = soup.select('td.title')
  
      for i in data:
          if i.a:
              # 리뷰
              review_dict['review'] = f_sibling(i.a, 5)
              review_dict_list.append(review_dict)
              review_dict = dict()
  
  def store(db, data):
      conn = sqlite3.connect(db)
      c = conn.cursor()
      c.execute('DROP TABLE IF EXISTS review')
      c.execute('''CREATE TABLE review(review text)''')
      c.executemany('INSERT INTO review VALUES(:review)', data)
      conn.commit()
      conn.close()
  
  def load_print(db):
      conn = sqlite3.connect(db)
      c = conn.cursor()
      c.execute('SELECT * FROM review')
      for i in c.fetchall():
          print(i)
  
  def f_sibling(n,x):
      for i in range(x):
          n=n.next_sibling
      return n.get_text().strip()
  
  # data 수집
  수집()
  
  # db 저장
  store('review.db',review_dict_list)
  
  # db 출력
  load_print('review.db')
  ```

  

# 2. CSV를 이용한 데이터 저장 및 활용

- 기본적인 사용(저장)

  ```python
  #1. 연결 : 기존 파일입출력에서 encoding과 newline을 다음과 같이 설정한다.
  f=open('ex.csv',"w",encoding='utf-8-sig',newline='')
  
  #2. writer 지정
  writer=csv.writer(f)
  
  #3. 첫번째 행은 제목으로 작성
  title = "제목1","제목2","제목3"
  writer.writerow(title)
  
  #4. 데이터 전달 및 입력 : 전달되는데이터는 list 자료형을 가져야한다.
  writer.writerows({데이터})
  ```

- 예시 : 네이버 뉴스에서 기사의 제목 및 소제목을 추출하여 CSV로 저장

  ```python
  import time
  from random import randint
  import csv
  import requests
  from bs4 import BeautifulSoup
  
  url = 'https://news.naver.com/main/list.naver?mode=LS2D&sid2=228&sid1=105&mid=shm&date=20220404&page='
  title = "제목", "내용"
  f = open('news.csv', "w", encoding='utf-8-sig', newline='')
  writer = csv.writer(f)
  writer.writerow(title)
  in_data = []
  
  for page in range(1,6):
      print(f"{page}page 크롤링중...")
      # 웹사이트의 PC접근을 위해서 헤더 설정
      r = requests.get(url+str(page),headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"})
      # 접속 상태 확인(접속 코드 200 아닐시 예외 발생)
      r.raise_for_status()  
      soup = BeautifulSoup(r.text, "html.parser")
      data = soup.select('dl')
      for i in data:
          for j in i.find_all('a'):
              if j.text.strip() != '':
                  in_data.append([j.text.strip(), i.span.text.strip()])
  	# 비정상적인 접근으로 인식되지 않기 위해서 시간을 3~10초로 설정
      time.sleep(randint(3,10))
  
  writer.writerows(in_data)
  f.close()
  ```

  