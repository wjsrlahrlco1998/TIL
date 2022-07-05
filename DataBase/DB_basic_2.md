# 기본 쿼리문 2



### 1) GROUP BY

```
-- Q1. 국가 별 고객수 조회(고객수 기준 오름차순) : 국가명, 고객수
SELECT Country, COUNT(Country)
FROM Customers
GROUP BY Country
ORDER BY COUNT(Country);

표준은 다음과 같이 GROUP BY을 적용했던 Columns만 SELECT문을 적용할 수 있다.

SELECT Country, COUNT(*)
FROM Customers
GROUP BY Country;
```

```
-- Q2. 국가 별, 도시 별 고객수 조회 (고객수 기준 내림차순) : 국가명, 도시명, 고객수
SELECT Country, City, COUNT(*)
FROM Customers
GROUP BY Country, City
ORDER BY COUNT(Country) DESC;
```

### 2) HAVING

- `GROUP BY` 사용후 필터링

```
-- Q1. 국가별 고객수를 조회하고 그 중 5명 초과인 국가만 조회 (고객수 내림 차순) : 국가명, 고객수
SELECT Country, COUNT(*)
FROM Customers
GROUP BY Country
HAVING COUNT(*) > 5
ORDER BY COUNT(*) DESC;

또는

SELECT Country, 고객수
FROM (
  SELECT Country, COUNT(*) AS 고객수
  FROM Customers
  GROUP BY Country
  )
WHERE 고객수 > 5
ORDER BY 고객수 DESC;

속도는 HAVING을 사용하는 것이 더 빠르다.
```

### 3) UNION

- 자주 사용되지 않음.
- 행 합침.

```
-- Q1. 고객의 국가와 도시 그리고 공급자의 국가와 도시를 모두 조회
SELECT Country, City
FROM Customers
UNION -- 중복 제거
SELECT Country, City
FROM Suppliers;

SELECT Country, City
FROM Customers
UNION ALL -- 중복을 제거하지 않음
SELECT Country, City
FROM Suppliers;
```

### 4) JOIN

- 열 기준 테이블 합침.

```
-- 카디즌 곱을 사용하는 방법
-- 테이블 조회를 하는데 제품 목록과 그에 해당하는 공급자 목록을 한 테이블에서 보고싶다
SELECT *
FROM Products, Suppliers
WHERE Products.SupplierID = Suppliers.SupplierID;
```

```
-- INNER JOIN 사용(디폴트가 INNER)
-- Q1. 상품을 조회하는데, 카테고리 이름과 함께 보이도록 조회 
SELECT *
FROM Products
	JOIN Suppliers
    ON Suppliers.SupplierID = Products.SupplierID;
```

```
-- Q2. 상품을 조회하는데, 카테고리 목록과 함께 보이도록 조회 
SELECT *
FROM Products
	INNER JOIN Categories
    ON Products.CategoryID = Categories.CategoryID
```

