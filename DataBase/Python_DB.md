# Python - DB 연동



## 1. DataBase 설치



### 1) MySQL 설치

- [Download](https://dev.mysql.com/downloads/)

  ![image-20221008213446463](Python_DB.assets/image-20221008213446463.png)

- [설치참고](https://suzxc2468.tistory.com/146)



### 2) HeidiSQL 설치

- SQL의 프론트 Tool

- [Download](https://www.heidisql.com/download.php)

  ![image-20221008212516352](Python_DB.assets/image-20221008212516352.png)




## 2. Python 연동



### 1) Package install

- ``pip install pymysql``

- ``pip install sqlalchemy`` 



### 2) Code



#### (1) import

```python
import pymysql
import pandas as pd
from sqlalchemy import create_engine
```



#### (2) DB Connection

```python
db_connection_str = 'mysql+pymysql://root:1234@localhost:3306/cases'
db_connection = create_engine(db_connection_str) 
conn = db_connection.connect()
```



#### (3) Data load

```python
df = pd.read_csv("./data/prec_data_refine.csv", encoding='utf-8-sig')
df.head()
```



#### (4) Insert DataFrame

```python
df.to_sql(name = 'cases_info',
         con=db_connection,
         if_exists='append',
         index=False)
```



#### (5) Read DataFrame

```python
read_df = pd.read_sql("SELECT * FROM cases_info", db_connection)
read_df = read_df.drop(['id'], axis=1)
read_df.head()
```

