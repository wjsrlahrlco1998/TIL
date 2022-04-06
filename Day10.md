# 1. 웹 컨트롤러-2

## 1) webdriver 추가 기능

- 스크롤

  - ``excute_script('window.scrollTo(0,document.body.scrollHeight)')``

    웹 창의 스크롤을 0부터 끝까지 내린다. 스크립트 명령어의 인자로 숫자를 주어서 조절할 수 있다.

  - 스크롤이 가변적인 웹 페이지의 스크롤을 끝까지 내리는 코드

    ```python
    while True:
        info_n = b.execute_script('return document.body.scrollHeight')#로드된 내용의 최하단 크기확인
        b.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        #확장
        time.sleep(2)
        next_n = b.execute_script('return document.body.scrollHeight')
        # 내리기 전의 최하단 크기와 내린 후의 최하단 크기가 같으면 페이지의 끝
        if info_n == next_n:
            break
    ```



## 2). selenium.webdriver 옵션지정

- 웹 컨트롤러의 옵션 지정

  - ``webdriver.ChromeOptions()`` : 웹 컨트롤러의 옵션을 설정할 객체를 반환
  - ``add_argument('window-size=1920x1080')`` : 지정된 크기로 페이지 오픈
  - ``add_argument('user-agent={user-agent 정보}')`` : 웹 컨트롤러에 user-agent 정보를 지정
  - ``headless`` : headless의 값을 True로 지정하면 페이지를 열지않고 실행한다.

  *웹 컨트롤러의 옵션은 **Headless 를 사용**하는 경우 지정하고, 그 외에는 사용자의 기본 브라우저의 옵션을 그대로 가져오기 때문에 설정하지 않아도된다.*

- 옵션 사용 예시

  ```python
  from selenium import webdriver
  from bs4 import BeautifulSoup
  
  # 웹 드라이버의 옵션을 설정
  op=webdriver.ChromeOptions()
  op.headless = True # 페이지를 열지않고 실행
  op.add_argument("window-size=1920x1080") # 윈도우 사이즈 설정
  op.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36')# user-agent의 값 설정
  b=webdriver.Chrome(options=op) # 옵션 부여
  b.get('URL')
  # 브라우저 종료 : 창이 백그라운드에서 실행되기 때문에 종료를 명시해주어야 한다.
  b.quit()
  
  ```




## 3) 옵션지정 및 스크롤을 활용한 동적 크롤링 예시

- headless를 활용하여 웹 창 없이 실행
- 크롤링 경로 : 네이버 -> 테슬라 검색 -> 뉴스 -> 제목, 내용 5page까지 크롤링
- .csv 파일로 저장 후 불러와서 출력

```python
from selenium import webdriver
from bs4 import BeautifulSoup
import csv

# csv 파일포인터 설정
f=open('tesla.csv','w',encoding='utf-8-sig',newline='')
writer=csv.writer(f)
csv_title="제목","내용"
writer.writerow(csv_title)
data_list=[]

op=webdriver.ChromeOptions()
op.headless = True
op.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36')

b=webdriver.Chrome(options=op)
b.implicitly_wait(10)
b.get('http://naver.com')
b.implicitly_wait(10)

# 테슬라 검색
b.find_element_by_xpath('//*[@id="query"]').send_keys('테슬라\n')
b.implicitly_wait(10)

# 뉴스 클릭
b.find_element_by_xpath('//*[@id="lnb"]/div[1]/div/ul/li[2]/a').click()
b.implicitly_wait(10)

# 데이터 추출
title_list=[]
content_list=[]
for page in range(2,7):
    b.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    b.implicitly_wait(10)
    html = b.page_source
    b.find_element_by_xpath(f'//*[@id="main_pack"]/div[2]/div/div/a[{page}]').click()
    b.implicitly_wait(10)
    s = BeautifulSoup(html,'html.parser')

    # 1번째 방법 : 분리 수집(단일로 접근)
    title = s.select('a.news_tit')
    content = s.select('a.api_txt_lines.dsc_txt_wrap')
    for i in title:
        title_list.append(i.text.strip())
    for i in content:
        content_list.append(i.text.strip())
    '''
    # 2번째 방법 : 복수로접근
    data=s.select('div.news_area')
    for i in data:
        print(i.select_one('a.news_tit').text)
        print(i.select_one('div.news_dsc').text)
    '''

data_list=list(zip(title_list,content_list))
writer.writerows(data_list)
f.close()


f=open('tesla.csv','r',encoding='utf-8-sig',newline='')
reader=csv.reader(f)
run=False
for i in reader:
    # 이 과정을 통해서 제목, 내용이 출력되지않는다.(즉, 첫번째 원소가 출력되지않는다.)
    if run:
        run=False
        continue
    print(f"제목:{i[0]}\n내용:{i[1]}\n")

b.quit()
```



