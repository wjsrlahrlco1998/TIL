# 웹 스크레핑이란?

- 웹 크롤링이라고도 한다
- 통신을 통해 웹 사이트에서 원하는 정보를 자동으로 수집하고 정리하는 것
- 반복적인 작업을 자동화 할 수 있다

## 1. Request를 이용한 크롤링

### 1) Request의 메소드

1. GET : 가져오기
2. POST : 붙이기(등록하기)
3. PUT : 수정하기
4. DELETE : 삭제하기

### 2) Request의 응답코드

1. 2xx : 성공
2. 3xx : 리다이렉션(자원이 옮겨짐)
3. 4xx : 요청 오류(우리 잘못)
4. 5xx : 서버 오류(서버 잘못)

### 3) 데이터 포맷 형식

1. XML
    - 태그 트리 형식
    - HTML로 XML의 일부
2. JSON
3. 바이너리 파일(이미지, 음악, 동영상, ...)

## 2. 크롤링의 순서

1. 적절한 요청을 보낸다.(Request)
    - 응답을 기다린다.
    - 응답을 받는다.
2. 응답을 분석한다.(Parsing)
3. 적절한 형태로 데이터를 저장한다.

## 3. Request : 요청 코드

### 1) 기본적인 요청 형식


```python
import requests
```


```python
response = requests.get('https://www.naver.com/') # get : 요청 보냄
response
```




    <Response [200]>




```python
response.headers # 헤더 확인
```




    {'Server': 'NWS', 'Date': 'Mon, 27 Jun 2022 01:33:01 GMT', 'Content-Type': 'text/html; charset=UTF-8', 'Transfer-Encoding': 'chunked', 'Connection': 'keep-alive', 'Set-Cookie': 'PM_CK_loc=5dc01a7faf8f5aaf85371b48623b591d361f79d4f1cc766370d92b9703bf49ce; Expires=Tue, 28 Jun 2022 01:33:01 GMT; Path=/; HttpOnly', 'Cache-Control': 'no-cache, no-store, must-revalidate', 'Pragma': 'no-cache', 'P3P': 'CP="CAO DSP CURa ADMa TAIa PSAa OUR LAW STP PHY ONL UNI PUR FIN COM NAV INT DEM STA PRE"', 'X-Frame-Options': 'DENY', 'X-XSS-Protection': '1; mode=block', 'Content-Encoding': 'gzip', 'Strict-Transport-Security': 'max-age=63072000; includeSubdomains', 'Referrer-Policy': 'unsafe-url'}




```python
response.content[:1000] # HTML 형식임을 확인
```




    b'\n<!doctype html>                          <html lang="ko" data-dark="false"> <head> <meta charset="utf-8"> <title>NAVER</title> <meta http-equiv="X-UA-Compatible" content="IE=edge"> <meta name="viewport" content="width=1190"> <meta name="apple-mobile-web-app-title" content="NAVER"/> <meta name="robots" content="index,nofollow"/> <meta name="description" content="\xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84 \xeb\xa9\x94\xec\x9d\xb8\xec\x97\x90\xec\x84\x9c \xeb\x8b\xa4\xec\x96\x91\xed\x95\x9c \xec\xa0\x95\xeb\xb3\xb4\xec\x99\x80 \xec\x9c\xa0\xec\x9a\xa9\xed\x95\x9c \xec\xbb\xa8\xed\x85\x90\xec\xb8\xa0\xeb\xa5\xbc \xeb\xa7\x8c\xeb\x82\x98 \xeb\xb3\xb4\xec\x84\xb8\xec\x9a\x94"/> <meta property="og:title" content="\xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84"> <meta property="og:url" content="https://www.naver.com/"> <meta property="og:image" content="https://s.pstatic.net/static/www/mobile/edit/2016/0705/mobile_212852414260.png"> <meta property="og:description" content="\xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84 \xeb\xa9\x94\xec\x9d\xb8\xec\x97\x90\xec\x84\x9c \xeb\x8b\xa4\xec\x96\x91\xed\x95\x9c \xec\xa0\x95\xeb\xb3\xb4\xec\x99\x80 \xec\x9c\xa0\xec\x9a\xa9\xed\x95\x9c \xec\xbb\xa8\xed\x85\x90\xec\xb8\xa0\xeb\xa5\xbc \xeb\xa7\x8c\xeb\x82\x98 \xeb\xb3\xb4\xec\x84\xb8\xec\x9a\x94"/> <meta name="twitter:card" content="summary"> <meta name="twitter:title" content=""> <meta name="twitter:url" content="https://www.naver.com/"> <meta name="twitter:image" content="https://s.pstatic.net/'



### 2) 조금 더 복잡한 요청 형식 : 쿼리 셋을 지정하여 요청


```python
response = requests.get('https://search.naver.com/search.naver', 
             params={'where' : 'news', 'query' : '무역전쟁'}) # 쿼리 셋(파라미터)를 지정하여 요청하기
response
```




    <Response [200]>




```python
response.content[:1000]
```




    b'<!doctype html> <html lang="ko"> <head> <meta charset="utf-8"> <meta name="referrer" content="always">  <meta name="format-detection" content="telephone=no,address=no,email=no"> <meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=2.0"> <meta property="og:title" content="\xeb\xac\xb4\xec\x97\xad\xec\xa0\x84\xec\x9f\x81 : \xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84 \xeb\x89\xb4\xec\x8a\xa4\xea\xb2\x80\xec\x83\x89"/> <meta property="og:image" content="https://ssl.pstatic.net/sstatic/search/common/og_v3.png"> <meta property="og:description" content="\'\xeb\xac\xb4\xec\x97\xad\xec\xa0\x84\xec\x9f\x81\'\xec\x9d\x98 \xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84 \xeb\x89\xb4\xec\x8a\xa4\xea\xb2\x80\xec\x83\x89 \xea\xb2\xb0\xea\xb3\xbc\xec\x9e\x85\xeb\x8b\x88\xeb\x8b\xa4."> <meta name="description" lang="ko" content="\'\xeb\xac\xb4\xec\x97\xad\xec\xa0\x84\xec\x9f\x81\'\xec\x9d\x98 \xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84 \xeb\x89\xb4\xec\x8a\xa4\xea\xb2\x80\xec\x83\x89 \xea\xb2\xb0\xea\xb3\xbc\xec\x9e\x85\xeb\x8b\x88\xeb\x8b\xa4."> <title>\xeb\xac\xb4\xec\x97\xad\xec\xa0\x84\xec\x9f\x81 : \xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84 \xeb\x89\xb4\xec\x8a\xa4\xea\xb2\x80\xec\x83\x89</title> <link rel="shortcut icon" href="https://ssl.pstatic.net/sstatic/search/favicon/favicon_191118_pc.ico">  <link rel="search" type="application/opensearchdescription+xml" href="https://ssl.pstatic.net/sstatic/search/opensearch-description.https.xml" title="Naver" /><link rel="stylesheet" type="text/css" href='




```python
response.text[:1000] # 한국어를 인코딩하여 출력
```




    '<!doctype html> <html lang="ko"> <head> <meta charset="utf-8"> <meta name="referrer" content="always">  <meta name="format-detection" content="telephone=no,address=no,email=no"> <meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=2.0"> <meta property="og:title" content="무역전쟁 : 네이버 뉴스검색"/> <meta property="og:image" content="https://ssl.pstatic.net/sstatic/search/common/og_v3.png"> <meta property="og:description" content="\'무역전쟁\'의 네이버 뉴스검색 결과입니다."> <meta name="description" lang="ko" content="\'무역전쟁\'의 네이버 뉴스검색 결과입니다."> <title>무역전쟁 : 네이버 뉴스검색</title> <link rel="shortcut icon" href="https://ssl.pstatic.net/sstatic/search/favicon/favicon_191118_pc.ico">  <link rel="search" type="application/opensearchdescription+xml" href="https://ssl.pstatic.net/sstatic/search/opensearch-description.https.xml" title="Naver" /><link rel="stylesheet" type="text/css" href="https://ssl.pstatic.net/sstatic/search/pc/css/search1_220602.css"> <link rel="stylesheet" type="text/css" href='




```python
resp = requests.get('https://www.wadiz.kr/web/wreward/ajaxGetCardList?startNum=48&limit=48&order=recommend&keyword=&endYn=ALL&utm_source=mkt_naver&utm_medium=search&utm_campaign=2022tvc_%EC%99%80%EB%94%94%EC%A6%88%EC%84%9C%ED%8F%AC%ED%84%B0_MO_%EC%8A%A4%ED%86%A0%EC%96%B4%EB%B8%8C%EB%9E%9C%EB%93%9C&utm_term=1.%EB%B8%8C%EB%9E%9C%EB%93%9C_%ED%8E%80%EB%94%A9&utm_content=WADIZ&gclid=CjwKCAjwh-CVBhB8EiwAjFEPGTTUc3AIoFVCMbLK1kJgJyTMLR8t5mHFTqeMg8BXN1wAbWa2G3xLmxoCVkgQAvD_BwE')
resp
```




    <Response [200]>




```python
resp.text[:1000] # JSON 형식임을 확인
```




    '{"success":"true","code":"SUSS000","title":null,"message":"","url":null,"additionalParams":{"totalCount":"39797"},"data":[{"campaignId":146752,"title":"음향장인 야마하 10만원대 사운드바ㅣ작은 사이즈에서 터지는 초강력 사운드","coreMessage":"\\"영상 속 소리가 현실이 됩니다.\\" 컴팩트 사이즈에서 나오는 100W 초강력 반전 사운드. 야마하 사운드바 SR-C20A 화이트 에디션이 모든 영상기기의 사운드 파트너가 됩니다!","photoUrl":"https://cdn.wadiz.kr/wwwwadiz/green001/2022/0512/20220512114323966_146752.jpg/wadiz/format/jpg/quality/80/optimize","photoThumbnail":0,"photoThumbnailUrl":null,"nickName":"(주)Tic2M","hostName":null,"encourageCnt":-1,"popularPoint":0,"participationCnt":788,"totalBackedAmount":136862000,"achievementRate":4562,"remainingDay":7,"isAllOrNothing":1,"endYn":0,"userPhotoUrl":null,"isOpen":1,"miniBoardCnt":0,"isStandingBy":0,"isSubmitted":0,"userId":-1,"encUserId":1026112008,"encIntUserId":1026112008,"globalId":0,"characterColor":null,"targetAmount":0,"targetMessage":null,"hashKeyword":null,"campaignUpdateCnt":0,"blurPhotoUrl":null,"blurCharacterColor":null,"custValueCode":287'




```python
resp.json() # JSON 형태 파싱 -> dict 형태로 바뀜
```




```python
data = resp.json()
data.keys()
```




    dict_keys(['success', 'code', 'title', 'message', 'url', 'additionalParams', 'data'])




```python
data['data']
```



## 4. Request : XML Parsing

### 1) BeautifulSoup


```python
from bs4 import BeautifulSoup
```


```python
resp = requests.get('https://naver.com')
resp
```




    <Response [200]>




```python
soup = BeautifulSoup(resp.text) # soup 객체 생성
```

#### * HTML TAGs
1. div
2. ul / ol => li
3. a -> (href)
4. img -> (scr)


```python
print(soup.prettify()) # 예쁘게 보이게 하기.
```


​    


```python
group_nav_tag = soup.find('div', class_='group_nav') # 태그와 클래스 명으로 찾기
group_nav_tag
```




    <div class="group_nav">
    <ul class="list_nav type_fix">
    <li class="nav_item">
    <a class="nav" data-clk="svc.mail" href="https://mail.naver.com/"><i class="ico_mail"></i>메일</a>
    </li>
    <li class="nav_item"><a class="nav" data-clk="svc.cafe" href="https://section.cafe.naver.com/">카페</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.blog" href="https://section.blog.naver.com/">블로그</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.kin" href="https://kin.naver.com/">지식iN</a></li>
    <li class="nav_item"><a class="nav shop" data-clk="svc.shopping" href="https://shopping.naver.com/"><span class="blind">쇼핑</span></a></li>
    <li class="nav_item"><a class="nav shoplive" data-clk="svc.shoppinglive" href="https://shoppinglive.naver.com/home"><span class="blind">쇼핑LIVE</span></a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.pay" href="https://order.pay.naver.com/home">Pay</a></li>
    <li class="nav_item">
    <a class="nav" data-clk="svc.tvcast" href="https://tv.naver.com/"><i class="ico_tv"></i>TV</a>
    </li>
    </ul>
    <ul class="list_nav NM_FAVORITE_LIST">
    <li class="nav_item"><a class="nav" data-clk="svc.dic" href="https://dict.naver.com/">사전</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.news" href="https://news.naver.com/">뉴스</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.stock" href="https://finance.naver.com/">증권</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.land" href="https://land.naver.com/">부동산</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.map" href="https://map.naver.com/">지도</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.vibe" href="https://vibe.naver.com/">VIBE</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.book" href="https://book.naver.com/">책</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.webtoon" href="https://comic.naver.com/">웹툰</a></li>
    </ul>
    <ul class="list_nav type_empty" style="display: none;"></ul>
    <a class="btn_more" data-clk="svc.more" href="#" role="button">더보기</a>
    <div class="ly_btn_area">
    <a class="btn NM_FAVORITE_ALL" data-clk="map.svcmore" href="more.html">서비스 전체보기</a>
    <a class="btn btn_set" data-clk="map.edit" href="#" role="button">메뉴설정</a>
    <a class="btn btn_reset" data-clk="edt.reset" href="#" role="button">초기화</a>
    <a class="btn btn_save" data-clk="edt.save" href="#" role="button">저장</a>
    </div>
    </div>




```python
type(group_nav_tag) # 데이터 타입 확인 : Tag
```




    bs4.element.Tag



#### find_all(), find()
- find_all() : 여러개 찾기 -> 데이터 타입은 태그의 리스트
- find : 하나 찾기 -> 데이터 타입은 태그


```python
group_nav_tags = soup.find_all('div', class_='group_nav')
type(group_nav_tags)
```




    bs4.element.ResultSet




```python
group_nav_tags[0]
```




    <div class="group_nav">
    <ul class="list_nav type_fix">
    <li class="nav_item">
    <a class="nav" data-clk="svc.mail" href="https://mail.naver.com/"><i class="ico_mail"></i>메일</a>
    </li>
    <li class="nav_item"><a class="nav" data-clk="svc.cafe" href="https://section.cafe.naver.com/">카페</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.blog" href="https://section.blog.naver.com/">블로그</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.kin" href="https://kin.naver.com/">지식iN</a></li>
    <li class="nav_item"><a class="nav shop" data-clk="svc.shopping" href="https://shopping.naver.com/"><span class="blind">쇼핑</span></a></li>
    <li class="nav_item"><a class="nav shoplive" data-clk="svc.shoppinglive" href="https://shoppinglive.naver.com/home"><span class="blind">쇼핑LIVE</span></a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.pay" href="https://order.pay.naver.com/home">Pay</a></li>
    <li class="nav_item">
    <a class="nav" data-clk="svc.tvcast" href="https://tv.naver.com/"><i class="ico_tv"></i>TV</a>
    </li>
    </ul>
    <ul class="list_nav NM_FAVORITE_LIST">
    <li class="nav_item"><a class="nav" data-clk="svc.dic" href="https://dict.naver.com/">사전</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.news" href="https://news.naver.com/">뉴스</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.stock" href="https://finance.naver.com/">증권</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.land" href="https://land.naver.com/">부동산</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.map" href="https://map.naver.com/">지도</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.vibe" href="https://vibe.naver.com/">VIBE</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.book" href="https://book.naver.com/">책</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.webtoon" href="https://comic.naver.com/">웹툰</a></li>
    </ul>
    <ul class="list_nav type_empty" style="display: none;"></ul>
    <a class="btn_more" data-clk="svc.more" href="#" role="button">더보기</a>
    <div class="ly_btn_area">
    <a class="btn NM_FAVORITE_ALL" data-clk="map.svcmore" href="more.html">서비스 전체보기</a>
    <a class="btn btn_set" data-clk="map.edit" href="#" role="button">메뉴설정</a>
    <a class="btn btn_reset" data-clk="edt.reset" href="#" role="button">초기화</a>
    <a class="btn btn_save" data-clk="edt.save" href="#" role="button">저장</a>
    </div>
    </div>



### 2) group_nav_tag의 a태그의 href 값 가져오기
- dictionary의 list
[
{
    'url': 'https://mail.naver.com',
    'name': '메일'
}, ...
]


```python
a_tags = group_nav_tag.find_all('a') # a 태그를 모두 가져오기(find_all 사용)
a_tags
```




    [<a class="nav" data-clk="svc.mail" href="https://mail.naver.com/"><i class="ico_mail"></i>메일</a>,
     <a class="nav" data-clk="svc.cafe" href="https://section.cafe.naver.com/">카페</a>,
     <a class="nav" data-clk="svc.blog" href="https://section.blog.naver.com/">블로그</a>,
     <a class="nav" data-clk="svc.kin" href="https://kin.naver.com/">지식iN</a>,
     <a class="nav shop" data-clk="svc.shopping" href="https://shopping.naver.com/"><span class="blind">쇼핑</span></a>,
     <a class="nav shoplive" data-clk="svc.shoppinglive" href="https://shoppinglive.naver.com/home"><span class="blind">쇼핑LIVE</span></a>,
     <a class="nav" data-clk="svc.pay" href="https://order.pay.naver.com/home">Pay</a>,
     <a class="nav" data-clk="svc.tvcast" href="https://tv.naver.com/"><i class="ico_tv"></i>TV</a>,
     <a class="nav" data-clk="svc.dic" href="https://dict.naver.com/">사전</a>,
     <a class="nav" data-clk="svc.news" href="https://news.naver.com/">뉴스</a>,
     <a class="nav" data-clk="svc.stock" href="https://finance.naver.com/">증권</a>,
     <a class="nav" data-clk="svc.land" href="https://land.naver.com/">부동산</a>,
     <a class="nav" data-clk="svc.map" href="https://map.naver.com/">지도</a>,
     <a class="nav" data-clk="svc.vibe" href="https://vibe.naver.com/">VIBE</a>,
     <a class="nav" data-clk="svc.book" href="https://book.naver.com/">책</a>,
     <a class="nav" data-clk="svc.webtoon" href="https://comic.naver.com/">웹툰</a>,
     <a class="btn_more" data-clk="svc.more" href="#" role="button">더보기</a>,
     <a class="btn NM_FAVORITE_ALL" data-clk="map.svcmore" href="more.html">서비스 전체보기</a>,
     <a class="btn btn_set" data-clk="map.edit" href="#" role="button">메뉴설정</a>,
     <a class="btn btn_reset" data-clk="edt.reset" href="#" role="button">초기화</a>,
     <a class="btn btn_save" data-clk="edt.save" href="#" role="button">저장</a>]




```python
sample = a_tags[0]
sample
```




    <a class="nav" data-clk="svc.mail" href="https://mail.naver.com/"><i class="ico_mail"></i>메일</a>




```python
# 문자열 가져오기
sample.text
```




    '메일'




```python
# class 속성의 값을 가져오기
sample.get('class')
```




    ['nav']




```python
# data-clk 속성의 값을 가져오기
sample.get('data-clk')
```




    'svc.mail'




```python
# a 태그의 href 값, url 값 가져오기
sample.get('href')
```




    'https://mail.naver.com/'



<목표> 
- 데이터를 원하는 형태로 만들기
1. 위 데이터가 있는 태그들을 모두 찾는다.
2. 태그들에서 url을 추출하고, text를 추출한다.
- ** text를 추출하는 함수는 <Tag>.text
- ** 속성을 추출하는 함수는 <Tag>.get()


```python
result = [{'url': tag.get('href'), 'name':tag.text} for tag in a_tags]
result
```




    [{'url': 'https://mail.naver.com/', 'name': '메일'},
     {'url': 'https://section.cafe.naver.com/', 'name': '카페'},
     {'url': 'https://section.blog.naver.com/', 'name': '블로그'},
     {'url': 'https://kin.naver.com/', 'name': '지식iN'},
     {'url': 'https://shopping.naver.com/', 'name': '쇼핑'},
     {'url': 'https://shoppinglive.naver.com/home', 'name': '쇼핑LIVE'},
     {'url': 'https://order.pay.naver.com/home', 'name': 'Pay'},
     {'url': 'https://tv.naver.com/', 'name': 'TV'},
     {'url': 'https://dict.naver.com/', 'name': '사전'},
     {'url': 'https://news.naver.com/', 'name': '뉴스'},
     {'url': 'https://finance.naver.com/', 'name': '증권'},
     {'url': 'https://land.naver.com/', 'name': '부동산'},
     {'url': 'https://map.naver.com/', 'name': '지도'},
     {'url': 'https://vibe.naver.com/', 'name': 'VIBE'},
     {'url': 'https://book.naver.com/', 'name': '책'},
     {'url': 'https://comic.naver.com/', 'name': '웹툰'},
     {'url': '#', 'name': '더보기'},
     {'url': 'more.html', 'name': '서비스 전체보기'},
     {'url': '#', 'name': '메뉴설정'},
     {'url': '#', 'name': '초기화'},
     {'url': '#', 'name': '저장'}]



### 3) CSS SELECTOR(선택자)
- soup.select("\<css selector\>")
    - class는 .
    - id는 #
    - 계층관계는 공백 or '>'


```python
data = soup.select('a.nav') # a 태그의 nav 클래스인 것 찾기
data
```




    [<a class="nav" data-clk="svc.mail" href="https://mail.naver.com/"><i class="ico_mail"></i>메일</a>,
     <a class="nav" data-clk="svc.cafe" href="https://section.cafe.naver.com/">카페</a>,
     <a class="nav" data-clk="svc.blog" href="https://section.blog.naver.com/">블로그</a>,
     <a class="nav" data-clk="svc.kin" href="https://kin.naver.com/">지식iN</a>,
     <a class="nav shop" data-clk="svc.shopping" href="https://shopping.naver.com/"><span class="blind">쇼핑</span></a>,
     <a class="nav shoplive" data-clk="svc.shoppinglive" href="https://shoppinglive.naver.com/home"><span class="blind">쇼핑LIVE</span></a>,
     <a class="nav" data-clk="svc.pay" href="https://order.pay.naver.com/home">Pay</a>,
     <a class="nav" data-clk="svc.tvcast" href="https://tv.naver.com/"><i class="ico_tv"></i>TV</a>,
     <a class="nav" data-clk="svc.dic" href="https://dict.naver.com/">사전</a>,
     <a class="nav" data-clk="svc.news" href="https://news.naver.com/">뉴스</a>,
     <a class="nav" data-clk="svc.stock" href="https://finance.naver.com/">증권</a>,
     <a class="nav" data-clk="svc.land" href="https://land.naver.com/">부동산</a>,
     <a class="nav" data-clk="svc.map" href="https://map.naver.com/">지도</a>,
     <a class="nav" data-clk="svc.vibe" href="https://vibe.naver.com/">VIBE</a>,
     <a class="nav" data-clk="svc.book" href="https://book.naver.com/">책</a>,
     <a class="nav" data-clk="svc.webtoon" href="https://comic.naver.com/">웹툰</a>]




```python
data = soup.select('div#NM_FAVORITE')
data
```




    <div class="gnb_inner" id="NM_FAVORITE">
    <div class="group_nav">
    <ul class="list_nav type_fix">
    <li class="nav_item">
    <a class="nav" data-clk="svc.mail" href="https://mail.naver.com/"><i class="ico_mail"></i>메일</a>
    </li>
    <li class="nav_item"><a class="nav" data-clk="svc.cafe" href="https://section.cafe.naver.com/">카페</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.blog" href="https://section.blog.naver.com/">블로그</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.kin" href="https://kin.naver.com/">지식iN</a></li>
    <li class="nav_item"><a class="nav shop" data-clk="svc.shopping" href="https://shopping.naver.com/"><span class="blind">쇼핑</span></a></li>
    <li class="nav_item"><a class="nav shoplive" data-clk="svc.shoppinglive" href="https://shoppinglive.naver.com/home"><span class="blind">쇼핑LIVE</span></a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.pay" href="https://order.pay.naver.com/home">Pay</a></li>
    <li class="nav_item">
    <a class="nav" data-clk="svc.tvcast" href="https://tv.naver.com/"><i class="ico_tv"></i>TV</a>
    </li>
    </ul>
    <ul class="list_nav NM_FAVORITE_LIST">
    <li class="nav_item"><a class="nav" data-clk="svc.dic" href="https://dict.naver.com/">사전</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.news" href="https://news.naver.com/">뉴스</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.stock" href="https://finance.naver.com/">증권</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.land" href="https://land.naver.com/">부동산</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.map" href="https://map.naver.com/">지도</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.vibe" href="https://vibe.naver.com/">VIBE</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.book" href="https://book.naver.com/">책</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.webtoon" href="https://comic.naver.com/">웹툰</a></li>
    </ul>
    <ul class="list_nav type_empty" style="display: none;"></ul>
    <a class="btn_more" data-clk="svc.more" href="#" role="button">더보기</a>
    <div class="ly_btn_area">
    <a class="btn NM_FAVORITE_ALL" data-clk="map.svcmore" href="more.html">서비스 전체보기</a>
    <a class="btn btn_set" data-clk="map.edit" href="#" role="button">메뉴설정</a>
    <a class="btn btn_reset" data-clk="edt.reset" href="#" role="button">초기화</a>
    <a class="btn btn_save" data-clk="edt.save" href="#" role="button">저장</a>
    </div>
    </div>
    <div class="group_weather" id="NM_WEATHER">
    <div>
    <a class="weather_area ico_w09" data-clk="squ.weat" href="https://weather.naver.com/today/08350107">
    <div class="current_box">
    <strong aria-label="현재기온" class="current">23.7°</strong><strong class="state">비</strong>
    </div>
    <div class="degree_box">
    <span aria-label="최저기온" class="min">23.0°</span><span aria-label="최고기온" class="max">26.0°</span>
    </div>
    <span class="location">좌동</span>
    </a>
    </div>
    <div>
    <a class="air_area" data-clk="squ.dust" href="https://weather.naver.com/today/08350107">
    <ul class="list_air">
    <li class="air_item">미세<strong class="state state_good">좋음</strong></li>
    <li class="air_item">초미세<strong class="state state_good">좋음</strong></li>
    </ul>
    <span class="location">좌동</span>
    </a>
    </div>
    </div>
    </div>




```python
# 공백 : 내부의 모든 항목 가져옴(자식 + 자손)
data = soup.select('.group_nav .nav_item') # div.group_nav li.nav_item 으로 써도된다.
data
```




    [<li class="nav_item">
     <a class="nav" data-clk="svc.mail" href="https://mail.naver.com/"><i class="ico_mail"></i>메일</a>
     </li>,
     <li class="nav_item"><a class="nav" data-clk="svc.cafe" href="https://section.cafe.naver.com/">카페</a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.blog" href="https://section.blog.naver.com/">블로그</a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.kin" href="https://kin.naver.com/">지식iN</a></li>,
     <li class="nav_item"><a class="nav shop" data-clk="svc.shopping" href="https://shopping.naver.com/"><span class="blind">쇼핑</span></a></li>,
     <li class="nav_item"><a class="nav shoplive" data-clk="svc.shoppinglive" href="https://shoppinglive.naver.com/home"><span class="blind">쇼핑LIVE</span></a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.pay" href="https://order.pay.naver.com/home">Pay</a></li>,
     <li class="nav_item">
     <a class="nav" data-clk="svc.tvcast" href="https://tv.naver.com/"><i class="ico_tv"></i>TV</a>
     </li>,
     <li class="nav_item"><a class="nav" data-clk="svc.dic" href="https://dict.naver.com/">사전</a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.news" href="https://news.naver.com/">뉴스</a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.stock" href="https://finance.naver.com/">증권</a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.land" href="https://land.naver.com/">부동산</a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.map" href="https://map.naver.com/">지도</a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.vibe" href="https://vibe.naver.com/">VIBE</a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.book" href="https://book.naver.com/">책</a></li>,
     <li class="nav_item"><a class="nav" data-clk="svc.webtoon" href="https://comic.naver.com/">웹툰</a></li>]




```python
# > : 나의 자식만 가져옴
data = soup.select('div.group_nav > li.nav_item')
data
```




    []



- find(하나만) vs find_all(여러개)
- select_one(하나만) vs select(여러개)


```python
soup.select_one('div.group_nav')
```




    <div class="group_nav">
    <ul class="list_nav type_fix">
    <li class="nav_item">
    <a class="nav" data-clk="svc.mail" href="https://mail.naver.com/"><i class="ico_mail"></i>메일</a>
    </li>
    <li class="nav_item"><a class="nav" data-clk="svc.cafe" href="https://section.cafe.naver.com/">카페</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.blog" href="https://section.blog.naver.com/">블로그</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.kin" href="https://kin.naver.com/">지식iN</a></li>
    <li class="nav_item"><a class="nav shop" data-clk="svc.shopping" href="https://shopping.naver.com/"><span class="blind">쇼핑</span></a></li>
    <li class="nav_item"><a class="nav shoplive" data-clk="svc.shoppinglive" href="https://shoppinglive.naver.com/home"><span class="blind">쇼핑LIVE</span></a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.pay" href="https://order.pay.naver.com/home">Pay</a></li>
    <li class="nav_item">
    <a class="nav" data-clk="svc.tvcast" href="https://tv.naver.com/"><i class="ico_tv"></i>TV</a>
    </li>
    </ul>
    <ul class="list_nav NM_FAVORITE_LIST">
    <li class="nav_item"><a class="nav" data-clk="svc.dic" href="https://dict.naver.com/">사전</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.news" href="https://news.naver.com/">뉴스</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.stock" href="https://finance.naver.com/">증권</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.land" href="https://land.naver.com/">부동산</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.map" href="https://map.naver.com/">지도</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.vibe" href="https://vibe.naver.com/">VIBE</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.book" href="https://book.naver.com/">책</a></li>
    <li class="nav_item"><a class="nav" data-clk="svc.webtoon" href="https://comic.naver.com/">웹툰</a></li>
    </ul>
    <ul class="list_nav type_empty" style="display: none;"></ul>
    <a class="btn_more" data-clk="svc.more" href="#" role="button">더보기</a>
    <div class="ly_btn_area">
    <a class="btn NM_FAVORITE_ALL" data-clk="map.svcmore" href="more.html">서비스 전체보기</a>
    <a class="btn btn_set" data-clk="map.edit" href="#" role="button">메뉴설정</a>
    <a class="btn btn_reset" data-clk="edt.reset" href="#" role="button">초기화</a>
    <a class="btn btn_save" data-clk="edt.save" href="#" role="button">저장</a>
    </div>
    </div>

