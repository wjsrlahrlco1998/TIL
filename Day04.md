# 1. 데이터 수집 및 관리



## 1) 웹 크롤링과 스크래핑

- 크롤링 : 웹 페이지에서 데이터를 가져오는 것
- 스크래핑 : 웹 페이지에서 원하는 데이터 일부를 가져오는 것

## 2) URL

-  정의

  웹에서 정해진 유일한 자원의 주소

## 3) Python 테이터 추출 방식

1. 인코딩 방식 추출 후 디코딩

   ```python
   import sys
   from urllib.request import urlopen
   
   # URL을 가리키는 포인터 f
   f = urlopen('URL')
   
   # URL의 인코딩 방식을 추출하고 없을 경우 utf-8로 지정
   encoding = f.info().get_content_charset(failobj="utf-8")
   
   # encoding의 방식으로 디코딩
   text=f.read().decode(encoding)
   ```

2. meta 태그에서 인코딩 방식 추출 후 디코딩

   ```python
   import re
   import sys
   from urllib.request import urlopen
   
   # URL을 가리키는 포인터 f
   f = urlopen('URL')
   
   # f로부터 읽은 데이터 저장
   bytes_content = f.read()
   
   # ASCII로 변환 : 정규표현식으로 나타내기 위함
   scanned_text = bytes_content[:1024].decode('ascii',errors='replace')
   
   # 정규표현식으로 charset 값을 추출
   match = re.search(r'charset=["\']?([\w-]+)',scanned_text)
   
   # charset이 존재하면 그것을 인코딩 방식으로 지정
   if match:
       encoding = match.group(1)
   else:
       # charset이 명시되지 않은 경우 utf-8로 지정
       encoding = 'utf-8'
   print('encoding:',encoding,file=sys.stderr)
   
   # 추출한 encoding으로 다시 디코딩
   text=bytes_content.decode(encoding)
   print(text)
   ```

   

# 2. 데이터 저장



## 1) 데이터의 저장방식

- txt : 데이터를 문자열로 변환후 텍스트 문자열로 저장
- pickle : 데이터를 이진 직렬 데이터로 변환 후 저장
  - 특징 : 데이터를 dump, load 하는데 있어서 자료형에 제한받지 않는다.
  - 단점 : 저장한 데이터를 확인하기 위해서는 코드로 load해야한다.
- json : 데이터를 문자열 형태로 저장
  - 특징 : 데이터를 dump, load 하는데 있어서 자료형에 제한받지 않고, 파일을 직관적으로 확인할 수 있다.
  - 단점 : 연속으로 dump하고 load하는데 제약을 받는다.
- csv : 데이터를 csv 형식의 파일로 저장
  - 특징 : 데이터를 csv 즉 Excel 형식으로 저장하여 가독성이 매우 뛰어나다.
  - 단점 : 데이터를 저장하고 불러오는데 많은 제약이 존재하며 구현이 복잡하다.
- SQL : SQL 언어를 기반한 DB로 저장 
  - 특징 : 데이터를 저장하는데 특화되어있다.
  - 단점 : 데이터를 이용하고자 하는데에는 제약이 따르며, SQL 언어를 알아야한다.



## 2) URL을 통한 데이터 추출 흐름

1. 데이터 추출

   ```python
   f=urlopen(url)
   encoding=f.info().get_content_charset(failobj="utf-8")
   html=f.read().decode(encoding)
   ```

2.  데이터 정규화(예시)

   ```python
   data=[]
   for i in re.findall(r'<td class="left"><a.*?</td>',html,re.DOTALL):
       url = re.search(r'<a href="(.*?)">',i).group(1)
       url="URL"+url
       title=re.sub(r'<.*?>','',i)
       title=unescape(title)
       data.append({'url':url,'title':title})
   ```

3. 저장(SQL이용)

   ```python
   conn = sqlite3.connect(db) # 1.경로연결
   c=conn.cursor() # 2. 커서지정
   # 3. 처리
   c.execute('DROP TABLE IF EXISTS books')
   c.execute('''
           CREATE TABLE books (
               title text,
               url text
           )
       ''')
   c.executemany('INSERT INTO books VALUES (:title, :url)',data)
   conn.commit() # 4. 저장
   conn.close() # 5. 종료
   ```

   

