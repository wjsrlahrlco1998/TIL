# 데이터 베이스 기본 3



### 1. INSERT, INTO

```sql
-- Q1. 고객 정보 추가
-- 1. column의 순서를 정해서 (모든 Column)
INSERT INTO Customers(CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country)
VALUES(92, '박지성', '삼성', '진구', '부산', '12312', 'Korea');

-- 2. column의 순서를 정하지 않고
INSERT INTO Customers
VALUES(93, '박지성', '삼성', '진구', '부산', '12312', 'Korea')

-- 3. column의 순서를 정하되 (customerID와 City, Country만 입력하기)
INSERT INTO Customers(CustomerID, City, Country)
VALUES(94, '서울', 'Korea');
```



### 2. DELETE

```sql
-- Q1. CustomerID가 92인 데이터 삭제
DELETE
FROM Customers
WHERE CustomerID=92;

-- Q2. 독일에 살고 있는 고객정보 삭제
DELETE
FROM Customers
WHERE Country='Germany';

WHERE 를 DELETE와 함께 사용시 해당 조건이 모두 삭제되기 때문에 사용에 유의해야한다.
또한 WHERE 을 사용하지 않으면 해당 테이블의 모든 데이터가 삭제되므로 유의해야한다.
```



### 3. UPDATE

```sql
-- Q1. 베를린(Berlin)에 살고 있는 고객의 우편번호를 12210 으로 수정
UPDATE Customers
SET PostalCode = '12210'
WHERE City='Berlin';
```



### 4. VIEW

```sql
CREATE VIEW [Brazil Customers] AS
SELECT CustomerName, ContactName
FROM Customers
WHERE Country = "Brazil";
```



### 5. 데이터 베이스 생성시 유의사항

- 참조 무결성(참조되는 키가 원래 테이블에서 삭제되는 경우)
  - 함께 삭제
  - null로 설정
  - 삭제를 못하게함

- 테이블 만들때 고려사항
  - 어떤 데이터들을 저장할지 결정(column)
  - 각 컬럼들이 어떤 데이터 타입을 가질지 결정
  - 각 컬럼들의 제약조건을 생각(null 허용 여부, 유일성, 등등)
    - Key를 설정 -> Primart key

