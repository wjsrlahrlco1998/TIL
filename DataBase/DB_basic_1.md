# SQL Query 문법

## 1. SQL Query

- `--` SQL의 주석
- 대.소문자 구분이 없으나 가독성을 위해서 표시한다.
- SQL문의 마지막에는 `;`을 붙인다.
- `*`는 Asterisk로 전체를 의미한다.
- `DISTINCT`는 유일값 즉, 중복을 제거한다.
- `LIKE '규칙'`는 특정 규칙을 정의한다.
- `NULL`은 대소구분이 없으므로 `IS NULL`, `IS NOT NULL`로 다룬다.
- ORDER BY에서 Default 값은 `ASC`이다.



### 1) SELECT

```
-- Q1. 고객(Customer)의 이름과 국가를 조회
SELECT CustomerName, Country
FROM Customers;
```

```
-- Q2. 고객(Customer) 정보 전체 조회
SELECT * FROM Customers;
```

```
-- Q3. 고객(Customer)의 국가 목록 조회(중복x)
SELECT DISTINCT Country
FROM Customers;
```

### 2) WHERE

```
-- Q1. 국가가 France 고객 조회
SELECT *
FROM Customers
WHERE Country='France';
```

```
-- Q2.ContactName이 'Mar'로 시작하는 고객 조회
SELECT *
FROM Customers
WHERE ContactName LIKE 'Mar%';
```

```
-- Q3. 이름이 'et'로 끝나는 직원 조회
SELECT *
FROM Employees
WHERE FirstName LIKE '%et';
```

### 3) AND, OR, NOT

```
-- Q1. 국가가 France이고 ContactName이 'Mar'로 시작하는 고객(Customers) 조회
SELECT *
FROM Customers
WHERE Country='France' AND ContactName LIKE 'Mar%';
```

```
-- Q2. 국가가 France이거나 ContactName이 'Mar'로 시작하는 고객 조회
SELECT *
FROM Customers
WHERE Country='France' OR ContactName LIKE 'Mar%'
```

```
-- Q3. 국가가 France가 아니고 ContactName이 'Mar'로 시작하지 않는 고객 조회
SELECT *
FROM Customers
WHERE NOT Country='France' AND ContactName NOT LIKE 'Mar%';

또는

WHERE Country<>'France' AND ContactName NOT LIKE 'Mar%';
```

### 4) IN, BETWEEN

```
-- Q1. 국가가 France 혹은 Spain에 사는 고객 조회
SELECT *
FROM Customers
WHERE Country IN ('France', 'Spain');
```

```
-- Q2. 가격이 15에서 20사이인 상품(Products) 조회
SELECT *
FROM Products
WHERE Price BETWEEN 15 AND 20;

또는 

Price >= 15 AND Price <= 20;
```

```
-- Q3. 가격이 15에서 20사이인 상품(Products)의 생산자 목록(SupplierName) 조회
SELECT DISTINCT SupplierName
FROM Suppliers
WHERE SupplierID IN (
	SELECT DISTINCT SupplierID -- SubQuery
	FROM Products
	WHERE Price BETWEEN 15 AND 20
    );
    
이것을 Query 안의 Query 즉 SubQuery라고 한다.
WHERE 뿐만 아니라 FROM, SELECT에도 들어갈 수 있다.
SubQuery의 차원에 따라 맞는 곳에 활용될 수 있다.
```

```
-- Q4. Price가 18인 Product의 Supplier목록 구하기
SELECT DISTINCT SupplierName
FROM Suppliers
WHERE SupplierID IN (
    SELECT SupplierID
    FROM Products
    WHERE Price=18
);
```

### 5) NULL

```
-- Q1. Phone이 NULL인 Shippers 조회
SELECT *
FROM Shippers
WHERE Phone IS NULL;

NULL이 아닌 것을 조회하려면 IS NOT NULL을 사용
```

### 6) ORDER BY

```
-- Q1. 상품 이름 오름차순(ASC)로 조회
SELECT *
FROM Products
ORDER BY ProductName ASC;
```

```
-- Q2. 상품 가격 내림차순(DESC)로 조회
SELECT *
FROM Products
ORDER BY Price DESC;
```

```
-- Q3. 상품 이름 오름차순(ASC)으로, 상품가격 내림차순(DESC)으로 조회
SELECT *
FROM Products
ORDER BY ProductName ASC, Price DESC;
```

### 7) TOP, LIMIT, ROWNUM - 표준이 존재하지 않음.

- MySQL 문법 예시

```
-- Q1. 국가가 UK 고객 중 이름순 3명 조회
SELECT *
FROM Customers
WHERE Country='UK'
ORDER BY CustomerName ASC
LIMIT 3;

여기서 OFFSET 3; 을 설정하면 시작위치를 3개 건너뛴 위치로 설정한다.
LIMIT과 OFFSET의 위치는 ORDER BY뒤에 설정한다.
```

### 8) CASE

- columns을 특정 조건에 맞추어 가공하여 새로운 columns을 만든다.

```
-- Q1. 상품 가격이 30미만, '저', 30~50 '중', 50 초과는 '고'로 조회
SELECT *,
	CASE
    	WHEN Price < 30 THEN '저'
        WHEN Price >= 30 AND Price <= 50 THEN '중'
        ELSE '고'
    END
FROM Products

SELECT *,
	(CASE
    	WHEN Price < 30 THEN '저'
        WHEN Price >= 30 AND Price <= 50 THEN '중'
        ELSE '고'
    END) AS 가격수준
FROM Products

로 사용하면 Columns의 이름을 바꿀 수 있다.
또한 AS는 생략 가능하다.
```

### 9) COUNT, AVG, SUM

```
-- Q1.France에 거주하는 고객수 조회 : 국가명, 고객수
SELECT Country, COUNT(*) AS 고객수
FROM Customers
WHERE Country='France';
```

```
-- Q2. 전체상품 평균가격 계산 : 평균가격
SELECT AVG(Price) AS 평균가격
FROM Products;
```

```
-- Q3. 주문 상품 수량 합계 계산: 주문 수량 합계
SELECT SUM(Quantity) AS 주문수량합계
FROM OrderDetails;
```

### 10) MIN, MAX

```
-- Q1. 상품 가격 중 최소값 조회 : 최소가격
SELECT MIN(Price) AS 최소가격
FROM Products;
```

```
-- Q2. 상품 가격 중 최대값 조회 : 최대가격
SELECT MAX(Price) AS 최대가격
FROM Products;
```

```
-- Q3.상품가격이 평균 이상인 상품들의 공급자 목록 조회
SELECT DISTINCT SupplierName
FROM Suppliers
WHERE SupplierID IN (SELECT SupplierID 
					FROM products 
                    WHERE Price >= (SELECT AVG(Price) FROM Products));
```



