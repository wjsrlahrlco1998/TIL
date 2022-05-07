# 1. 웹 컨트롤러

- 정의

  사람이 웹을 사용하는 것처럼 사용할 수 있도록 해주는 모듈

- 준비(Chrome)

  1. 사용하는 브라우저의 버전에 맞는 웹드라이버를 다운받고 인터프리터의 위치에 저장한다.

  2. selenium 모듈을 install한다.

     ``pip install selenium``

- 코드

  - 모듈 import

    ```python
    # 웹 컨트롤러
    from selenium import webdriver
    # I/O의 입력을 나타내는 명령어
    from selenium.webdriver.common.keys import Keys
    # 크롤링 차단을 회피하기 위한 모듈
    import pyperclip
    ```

  - 코드흐름

    ```python
    #1. 브라우저 접속
    b=webdriver.Chrome()
    b.get('URL')
    
    #2. 원하는 요소의 위치(여러방법이 존재)
    lc=b.find_element_by_xpath('xpath')
    lc=b.find_element_by_class_name('classname')
    
    #3. 요소를 클릭
    lc.click()
    
    #4. I/O(Ctrl+v)
    lc.send_keys(Keys.CONTROL,'v')
    
    #5. IP Ban을 방지하기 위한 Term
    b.implicitly_wait(10)
    ```

  - 캡쳐보드 사용

    ```python
    # 캡쳐보드로 복사했던 데이터를 요소에 붙여넣기로 입력
    in_id=b.find_element_by_xpath('xpath')
    in_id.click()
    pyperclip.copy(id)
    in_id.send_keys(Keys.CONTROL,'v')
    ```

    *캡쳐보드를 사용하여 봇 차단을 회피*

- 예시(네이버-로그인-뉴스-IT/과학-IT일반)

  ```python
  from selenium import webdriver
  from selenium.webdriver.common.keys import Keys
  import pyperclip
  
  id='your id'
  pw='your pw'
  
  # 브라우저 접속
  b=webdriver.Chrome()
  b.get('http://naver.com')
  
  b.implicitly_wait(10)
  
  # 로그인창 접속
  lc=b.find_element_by_xpath('//*[@id="account"]/a')
  lc.click()
  
  b.implicitly_wait(10)
  
  # id 입력
  in_id=b.find_element_by_xpath('//*[@id="id"]')
  in_id.click()
  pyperclip.copy(id)
  in_id.send_keys(Keys.CONTROL,'v')
  
  b.implicitly_wait(10)
  
  # pw 입력
  in_pw=b.find_element_by_xpath('//*[@id="pw"]')
  in_pw.click()
  pyperclip.copy(pw)
  in_pw.send_keys(Keys.CONTROL,'v')
  
  b.implicitly_wait(10)
  
  # 로그인
  in_login=b.find_element_by_xpath('//*[@id="log.login"]')
  in_login.click()
  
  b.implicitly_wait(10)
  
  # 뉴스
  in_news=b.find_element_by_xpath('//*[@id="NM_FAVORITE"]/div[1]/ul[2]/li[2]/a')
  in_news.click()
  
  b.implicitly_wait(10)
  
  # IT/과학
  in_IT=b.find_element_by_xpath('/html/body/section/header/div[2]/div/div/div[1]/div/div/ul/li[6]/a/span')
  in_IT.click()
  
  b.implicitly_wait(10)
  
  # IT 일반
  in_N_IT=b.find_element_by_xpath('//*[@id="snb"]/ul/li[4]/a')
  in_N_IT.click()
  
  b.implicitly_wait(10)
  ```

  

