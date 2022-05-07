# 1. 판다스를 활용한 금융데이터 수집 예제

## 1) 문제

 1. 동적 크롤링을 이용하여 데이터 수집

    수집경로 : 네이버 - '금융'검색 - 네이버 금융(클릭) - 시장지표(클릭) - 국제시장환율(1~5page)

 2. 통화명, 심볼명, 현재가, 전일대비, 등락율 스크래핑 후 .csv 파일로 저장하라.

 3. pandas를 이용하여 데이터테이블을 구성하라.



## 2) 코드

1. **import**

   ```python
   from selenium import webdriver
   from bs4 import BeautifulSoup
   import pandas as pd
   from pandas import DataFrame
   import numpy as np
   import csv
   ```

   

2. 데이터 수집

   ```python
   #1.웹 오픈
   b=webdriver.Chrome()
   b.get('http://naver.com')
   b.implicitly_wait(10)
   
   #2.'금융' 검색
   b.find_element_by_xpath('//*[@id="query"]').send_keys('금융\n')
   b.implicitly_wait(10)
   
   #3.네이버 금융 클릭
   #화면 아래에 존재하므로 스크롤을 내린다
   b.execute_script('window.scrollTo(0,500)')
   b.implicitly_wait(10)
   b.find_element_by_xpath('//*[@id="web_1"]/div/div[2]/div[2]/a').click()
   b.implicitly_wait(10)
   
   #4.시장지표 클릭
   # 새 브라우저가 열렸을때 스위칭이 필요하다.(0: 기존, 1: 새탭)
   b.switch_to.window(b.window_handles[1]) 
   b.find_element_by_xpath('//*[@id="menu"]/ul/li[4]/a/span').click()
   b.implicitly_wait(10)
   
   #5.국제시장 환율 클릭
   b.execute_script('window.scrollTo(0,700)')
   b.implicitly_wait(10)
   b.find_element_by_xpath('//*[@id="tab_section"]/ul/li[2]/a/span').click()
   b.implicitly_wait(10)
   #중요!! 해당 형태는 내부 inframe 형태로 html 속의 html이다. 따라서 해당 frame으로 스위칭 해주어야 html 데이터를 가져올 수 있다.
   b.switch_to.frame('frame_ex2') 
   
   #6. 데이터 추출-1 (각각의 데이터를 추출하는 방법_비효율적방법)
   통화명_list=[]
   심볼명_list=[]
   가격_list=[]
   for page in range(2,6):
       b.execute_script('window.scrollTo(0,document.body.scrollHeight)')
       b.implicitly_wait(10)
       html = b.page_source
       b.find_element_by_xpath(f'/html/body/div/div/a[{page}]').click()
       b.implicitly_wait(10)
       s = BeautifulSoup(html, 'html.parser')
       # 통화명 수집
       통화명=s.select('td.tit')
       for i in 통화명:
           통화명_list.append(i.a.text)
   
       # 심볼명 수집
       심볼명=s.select('td.symbol')
       for i in 심볼명:
           심볼명_list.append(i.a.text)
   
       # 가격 수집
       가격=s.select('td.num')
       모음=[]
       count=1
       for i in 가격:
           모음.append(i.text.strip().replace('\n','').replace('												','').replace('						 ',''))
           if count % 3 == 0:
               모음=[]
               가격_list.append(모음)
           count+=1
   현재가_list=[]
   전일대비_list=[]
   등락율_list=[]
   
   for i in 가격_list:
       if len(i) > 0:
           현재가_list.append(i[0])
           전일대비_list.append(i[1])
           등락율_list.append(i[2])
   
   data_list=list(zip(통화명_list,심볼명_list,현재가_list,전일대비_list,등락율_list))
   
   #6. 데이터 추출-2(매우 효율적인 방법)
   # 데이터를 정리
   def f(x):
       return x.text.strip().replace("\t", "").replace("\n", "")
   
   data = s.select('tr')
   data_list=[]
   for i in data:
       data_list.append(list(map(f,i.select('td'))))
   ```

   

3. 데이터 저장

   ```python
   # csv 파일로 저장
   f=open('금융.csv','w',encoding='utf-8-sig',newline='')
   writer=csv.writer(f)
   csv_title="통화명","심볼명","현재가","전일대비","등락율"
   writer.writerow(csv_title)
   writer.writerows(data_list)
   f.close()
   ```

4. pandas 데이터프레임 생성

   ```python
   # pandas 데이터프레임 생성
   df = pd.read_csv('금융.csv',encoding='utf-8-sig')
   df = DataFrame(df)
   print(df)
   ```

5. 데이터 등락율을 기준으로 증가, 감소로 각각의 프레임으로 나누기

   ```python
   # 문자열의 비교연산
   # 인덱스를 제거하지 않으면 원본데이터의 인덱스가 그대로 붙여져서 나온다.
   # 문자열 > 비교문자 : 비교문자가 들어있는 경우 True || 단 공백이 있으면 True로 반환하기 때문에 제외해야한다.
   # 문자열 < 비교문자 : 비교문자가 없는경우 True
   # 다른 테이블을 통해 새 테이블을 만들었다면 reset_index()를 통해서 index를 재설정해야한다.
   t1=df[(df['등락율']>'-') & (df['등락율'] != '0.00%')].reset_index(drop='index')
   print(t1)
   t2=df[df['등락율']<'-'].reset_index(drop='index')
   print(t2)
   ```

   - '5'에서 사용한 데이터를 나누는 방법은 매우 중요하다.
     - pandas의 데이터 형태가 numpy라는 것을 이용하는것
     - 별도의 형변환이 필요없이 조건에 따른 추출이 가능하다.