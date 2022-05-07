# 1. 웹 크롤링 예시(HTML)

```python
import requests
from bs4 import BeautifulSoup

url = 'https://finance.naver.com/sise/sise_rise.naver?sosok=1'
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
r_data = soup.select('td')
cos_dic = dict()

# 데이터의 나열 특성을 이용하여 추출
def number_data(n,x):
    for i in range(x):
        n=n.next_sibling

    return n.get_text()

# 데이터 추출
for i in r_data:
    if i.a:
        # 종목명 추출
        cos_dic[i.a.get_text()]=[]
        # 가격 추출
        for j in range(0, 21, 2):
            cos_dic[i.a.get_text()].append(number_data(i,j).strip())

# 선택 종목 정보 출력
name = input("종목명:")
try:
    print(f"종목명:{name} 현재가:{cos_dic[name][0]} 전일비:{cos_dic[name][1]} 등락률:{cos_dic[name][2]} 거래량:{cos_dic[name][3]}"
          f" 매수호가:{cos_dic[name][4]} 매도호가:{cos_dic[name][5]} 매수총잔량:{cos_dic[name][6]} 매도총잔량:{cos_dic[name][7]} PER:{cos_dic[name][8]}"
          f" ROE:{cos_dic[name][9]}")
except:
    print("해당 종목은 존재하지 않습니다.")
```



- 네이버-금융-코스닥 에서 종목명 및 가격에 대한 정보들을 추출하여 사전형으로 변환 후 선택 종목에 대한 정보를 제공하는 예시 프로그램