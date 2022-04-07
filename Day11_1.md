# 1. Pandas 

## 1) 정의

- 파이썬의 데이터 분석 라이브러리
  - 데이터 테이블을 다룸
- 기본적으로 numpy를 사용

## 2) 용어

- 데이터프레임(DataFrame)
  - 데이터 테이블 전체 객체(2차원)
- 시리즈(Series)
  - 각 열 데이터를 다루는 객체(1차원)

## 3) Series 객체

### (1) 정의

- 하나의 피쳐 데이터를 포함하는 형태

### (2) 세가지 요소

- 데이터
- 인덱스
- 데이터 타입

### (3) 생성 및 기능

- import
  - 기본적으로 pandas 안에 존재하나 편의를 위해 다음과 같이 선언함
  - ``from pandas import Series, DataFrame``

- 생성

  ```python
  data_l=[1,2,3,4,5]
  # 인덱스 지정 x -> 0부터 차례대로 부여
  e_obj=Series(data_l)
  
  '''
  결과:
  0    1
  1    2
  2    3
  3    4
  4    5
  dtype: int64
  '''
  
  # 인덱스 지정
  idx_l=[1,2,3,4,1]
  e_obj=Series(data=data_l,index=idx_l)
  # 키 값은 중복될 수 있으나 별 개의 키워드이다
  # 또한 키 값은 문자열형태로 지정될 수 있다
  '''
  결과:
  1    1
  2    2
  3    3
  4    4
  1    5
  dtype: int64
  '''
  
  # 객체 데이터 : 가능하다
  e_obj=Series(data=[data_l,1,2,3,4],index=idx_l)
  
  '''
  결과:
  a    [1, 2, 3, 4, 5]
  b                  1
  c                  2
  d                  3
  e                  4
  dtype: object
  ```

- 객체의 이름 변경

  - ``e_obj.name="이름"``

    지정 이름으로 객체의 이름을 지정한다.

- 인덱스 열의 이름 지정

  - ``e_obj.index.name="이름"``

    지정 이름으로 인덱스 열의 이름을 지정한다

- dict 자료형을 시리즈 객체로 변환

  ```python
  dic={'a':1,'b':2,'c':3,'d':4,'e':5,'g':6}
  e_obj=Series(dic,dtype=np.float32,name='data_s') 
  
  '''
  결과:
  a    1.0
  b    2.0
  c    3.0
  d    4.0
  e    5.0
  g    6.0
  Name: data_s, dtype: float32
  '''
  ```

- 판다스의 모든 객체는 인덱스 값을 기준으로 생성되고, 기존 데이터에 인덱스 값을 추가하면 NaN 값이 출력된다. 또한 dic의 키가 index와 일치하지 않는경우 무시된다.

  ```python
  idx_d=['a','b','c','d','e','f','h']
  e_obj=Series(dic,index=idx_d) # 키의 수 > 값의 수 인 경우에는 시리즈가 생성된다.
  
  '''
  결과:
  a    1.0
  b    2.0
  c    3.0
  d    4.0
  e    5.0
  f    NaN
  h    NaN
  dtype: float64
  '''
  ```

  

## 4) 데이터프레임

### (1) 정의

- 데이터 테이블 전체를 지칭하는 객체
- numpy 배열의 특성을 그대로 가진다
- 인덱싱 : 열과 행 각각 사용하여 하나의 데이터에 접근한다

### (2) 데이터 프레임 생성

- 데이터 프레임 직접 생성

  - ``e_obj=DataFrame(data,index=['1','2',...])``

    data는 dict 자료형

  - ``e_obj.reset_index(옵션)``

    - 옵션 
      - drop=True : 제거
      - inplace=True : 원본 반영

  - ``e_obj.set_index('{키}',inplace=True)``

    인덱스를 해당 키 값으로 전환한다.

  - ``e_obj.sort_index(ascending=False)``

    - ascending=False 는 인덱스를 내림차순 정렬
    - True는 오름차순 정렬이다

  - 예시코드

    ```python
    data={
        '이름':["홍길동","도우너","희동이"],
        '계좌번호':['1234','4321','4567'],
        '금액':[10000,100,500]
    }
    e_obj=DataFrame(data,index=['1번고객','2번고객','3번고객'])
    
    '''
    결과:
    
    			이름	계좌번호	금액
    1번고객	홍길동	1234	10000
    2번고객	도우너	4321	100
    3번고객	희동이	4567	500
    '''
    ```

    

- read_확장자로 데이터 로딩

  - csv파일 로딩

  ```python
   # csv 데이터 가져오기 // header=None : 타이틀 없이한다.->자동부여
  df_data=pd.read_csv('housing.data',sep='\s+',header=None
  df=DataFrame(df_data)             
  ```

- 파일 생성 및 저장

  - csv 파일로 저장

    ```python
    df.to_csv('data_1.csv',encoding='utf-8-sig') # 데이터프레임을 csv 파일로 저장 (index를 포함하여 저장)
    ```

  - 텍스트(.txt)저장

    ```python
    # sep는 구분자로 불러올때도 똑같이 지정해야한다
    df.to_csv('data3.txt',sep='\t')
    ```

  - 엑셀파일 저장

    ```python
    df.to_excel("data4.xlsx",index=False)
    ```

- 파일 열기

  - csv,txt,excel 파일 열기

    ```python
    # 기본 열기
    df=pd.read_csv('data_1.csv')
    
    # skiprows와 header 옵션 이용 열기
    # skiprows = n : 처음부터 n번째 라인을 읽지 않는다. 리스트로 지정가능.
    # header=None 은 헤더를 지정하지 않고 연다.
    df=pd.read_csv('data_2.csv',skiprows=1,header=None)
     # 1, 3번째 데이터를 제외하고 가져온다.
    df=pd.read_csv('data_2.csv',skiprows=[1,3],header=None)
    
    df=pd.read_csv('data_2.csv',skiprows=[1,3],nrows=2) # 1,3번째 데이터를 제외하고 2개의 데이터만 가져오겠다.
    
    # 텍스트 파일도 read_csv로 읽어온다 
    df=pd.read_csv('data3.txt',sep='\t')
    #index_col = '속성' 을 인덱스로 설정
    df.set_index("이름",inplace=True)
    
    # '이름'을 인덱스로 파일 열기
    df=pd.read_excel('data4.xlsx', index_col='이름')
    ```

- 여러 기능

  - ``df_t[['이름','은행']][:2]``

    슬라이싱을 통한 데이터 추출 가능

  - ```python
    # 데이터 가변 : 인덱스를 이름으로 사용
    df_t.index=df_t['이름']
    
    # 제거
    del df_t['이름']
    
    # loc : 인덱스 이름과 열 이름으로 데이터 추출
    # 원하는 대상의 원하는 값만 추출 가능 (고차원:세로줄, 저차원:가로줄)
    df_t.loc[['홍길동','희동이'],['금액']] 
    
    # 키 값으로 인덱스로 설정가능 + 슬라이싱 가능
    df_t.loc['도우너':,['금액']] 
    
    # iloc : 인덱스 번호로만 데이터 호출
    df_t.iloc[:2,:2]
    ```

  - head, tail

    처음 n개 행, 마지막 n개 행 호출

    ```python
    df.head(n)
    df.tail(n)
    ```