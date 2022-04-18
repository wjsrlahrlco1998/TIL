# 1. 크롤링 - 데이터 저장 및 불러오기 - wordcloud

- 방법

  1. 네이버 접속
  2. 빅데이터 검색
  3. 뉴스 접속
  4. 1~5 page 제목 추출
  5. csv 파일로 저장
  6. csv 읽기 후 dataframe 변환
  7. wordcloud로 표현

- 코드

  ```python
  import pandas as pd
  from selenium import webdriver
  from bs4 import BeautifulSoup
  from wordcloud import WordCloud, STOPWORDS
  import matplotlib.pyplot as plt
  from collections import Counter
  from konlpy.tag import Okt
  
  # 공백제거함수
  def del_space(x):
      return x.text.strip().replace("\t", "").replace("\n", "").replace(',', "")
  
  # 네이버 접속
  b = webdriver.Chrome()
  b.get('http://naver.com')
  b.implicitly_wait(10)
  
  # 빅데이터 검색
  b.find_element_by_xpath('//*[@id="query"]').send_keys('빅데이터\n')
  b.implicitly_wait(10)
  
  # 뉴스 클릭
  b.find_element_by_xpath('//*[@id="lnb"]/div[1]/div/ul/li[4]/a').click()
  b.implicitly_wait(10)
  
  # 제목 추출
  title_list = []
  for page in range(2, 6):
      html = b.page_source
      s = BeautifulSoup(html, 'html.parser')
  
      # 페이지 이동
      b.execute_script('window.scrollTo(0,document.body.scrollHeight)')
      b.implicitly_wait(10)
      b.find_element_by_xpath(f'//*[@id="main_pack"]/div[2]/div/div/a[{page}]').click()
      b.implicitly_wait(10)
  
      data = s.select('a.news_tit')
      for i in data:
          if i:
              title_list.append(del_space(i))
  
  # 데이터 저장
  df = pd.DataFrame(title_list, columns=['제목'])
  df.to_csv('Bigdata_news_title.csv', encoding='utf-8')
  
  # 데이터 불러오기
  df = pd.read_csv('Bigdata_news_title.csv', encoding='utf-8')
  
  # wordcloud로 표현할 데이터 추출
  title = df['제목']
  
  # 형태소 분석을 통해서 명사로 분류
  ok_t = Okt()
  nounData_list = []
  for i in title:
      t = ok_t.nouns(i)
      nounData_list.extend(t)
  
  # 한글자 단어들은 제거
  filter_len_one = [i for i in nounData_list if len(i) > 1]
  
  # 각 단어 마다의 빈도수로 변환
  bin_data = Counter(filter_len_one)
  
  # 빈도수 별로 내림차순 정렬 후 상위 100개의 데이터만 가져옴
  sorted_bin_data = bin_data.most_common(100)
  
  # 가져온 데이터를 dict 형으로 형 변환
  dict_data = dict(sorted_bin_data)
  
  # Wordcloud로 표현
  wc = WordCloud(font_path='C:\Windows\Fonts\malgun.ttf', background_color='white')
  g_data = wc.generate_from_frequencies(dict_data)
  plt.figure(figsize=(10,10))
  plt.axis('off')
  plt.imshow(g_data)
  plt.show()
  ```

- 실행결과

  ![image-20220418204137053](Day15_4.assets/image-20220418204137053.png)

  