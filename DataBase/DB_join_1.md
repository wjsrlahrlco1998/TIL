# SQL JOIN 문 실습 문제



```
-- Q1. Londona에 위치한 공급자(Supplier)가 생산한 상품 목록 조회 : 도시명, 공급자명, 상품명, 상품가격
SELECT City, SupplierName, ProductName, Price
FROM Suppliers S
	JOIN Products P
    ON S.SupplierID = P.SupplierID
WHERE S.City LIKE 'Londona';
```

```
-- Q2. 분류가 Seafood 인 상품 목록 (모든 컬럼 조회) : 분류, 상품 모든 컬럼
SELECT *
FROM Products P
	JOIN Categories C
    ON P.CategoryID = C.CategoryID
WHERE CategoryName LIKE 'Seafood';
```

```
-- Q3. 공급자(Supplier) 국가별, 카테고리 별 상품 건수, 평균가격 : 국가명, 카테고리명, 상품건수, 평균가격
SELECT Country, CategoryName, COUNT(*), AVG(Price)
FROM (
  SELECT *
  FROM Suppliers S
      JOIN Products P
      ON S.SupplierID = P.SupplierID
      ) SP
      JOIN Categories C
      ON SP.CategoryID = C.CategoryID
GROUP BY Country, CategoryName
```

```
-- Q4. 상품 카테고리별, 국가별, 도시별 주문건수 2건 이상인 목록 : 카테고리명, 국가명, 도시명, 주문건수 (공급자)
SELECT CategoryName, Country, City, COUNT(CategoryName) AS 주문건수
FROM (
  SELECT *
  FROM Products P
      JOIN Categories C
      ON P.CategoryID = C.CategoryID
      ) PC
      JOIN Suppliers S
      ON PC.SupplierID = S.SupplierID 
GROUP BY CategoryName, Country, City
HAVING COUNT(CategoryName) >= 2
```

```
-- Q5. 주문자, 주문정보, 직원정보, 배송자정보 통합 조회 : 고객컬럼 전체, 주문정보 컬럼 전체(order, orderDetail), 배송자 정보 컬럼 전체 (4개 테이블 조인)
SELECT *
FROM (
  SELECT *
  FROM (
    SELECT *
    FROM (
      SELECT *
      FROM Orders O
          JOIN Employees E
          ON O.EmployeeID = E.EmployeeID
          ) OE
          JOIN Customers C
          ON OE.CustomerID = C.CustomerID
      ) OEC
      JOIN OrderDetails OrD
      ON OEC.OrderID = OrD.OrderID
	) OECOrD 
    JOIN Shippers S 
    ON OECOrD.ShipperID = S.ShipperID;
```

```
-- Q6. 판매량(Quantity) 상위 TOP 3 공급자(supplier) 목록 : 공급자 명, 판매량, 판매금액 (판매량의 합이 아닌, 판매량이 가장 높은 상품)
SELECT ShipperName, SUM(Quantity), Price
FROM (
  SELECT *
  FROM (
    SELECT *
    FROM Shippers S
        JOIN Orders O
        ON S.ShipperID = O.ShipperID
        ) SO
        JOIN OrderDetails Od
        ON SO.OrderID = Od.OrderID
        ) SOOd
        JOIN Products P
        ON SOOd.ProductID = P.ProductID
GROUP BY ShipperName
ORDER BY SUM(Quantity) DESC;
```

```
-- Q7. 상품(Product) 분류(Category)별, 고객 지역별(City) 판매량 순위별 정렬 : 카테고리명, 고객지역명, 판매량 (판매량의 합이 아닌, 판매량이 높은 상품)
SELECT CategoryName, City, Quantity
FROM (
  SELECT *
  FROM (
    SELECT *
    FROM (
      SELECT *
      FROM Customers C
          JOIN Orders O
          ON C.CustomerID = O.CustomerID
          ) CO
          JOIN OrderDetails Ord
          ON CO.OrderID = Ord.OrderID
          ) COOrd
          JOIN Products P
          ON COOrd.ProductID = P.ProductID
          ) COOrdP
          JOIN Categories C
          ON COOrdP.CategoryID = C.CategoryID
GROUP BY CategoryName, City
ORDER BY Quantity DESC;
```

```
-- Q8. 고객 국가가 USA이고, 상품별 판매량(Quantity 수량 합계) 순위별 정렬 : 국가명, 상품명, 판매량, 판매금액 
SELECT Country, ProductName, SUM(Quantity), Price
FROM (
  SELECT *
  FROM (
    SELECT *
    FROM (
      SELECT *
      FROM Customers
      WHERE Country LIKE 'USA'
      ) C
      JOIN Orders O
      ON C.CustomerID = O.CustomerID
      ) CO
      JOIN OrderDetails Ord
      ON CO.OrderID = Ord.OrderID
      ) COOrd
      JOIN Products P
      ON COOrd.ProductID = P.ProductID
GROUP BY ProductName
ORDER BY SUM(Quantity) DESC;
```

```
-- Q9. Supplier의 국가가 Germany인 상품 카테고리별 상품 수 : 국가명, 카테고리명, 상품수
SELECT Country, CategoryName, COUNT(ProductName)
FROM (
  SELECT *
  FROM (
    SELECT *
    FROM Suppliers
    WHERE Country LIKE 'Germany'
    ) S
    JOIN Products P
    ON P.SupplierID = S.SupplierID
    ) SP
    JOIN Categories C
    ON SP.CategoryID = C.CategoryID;
```

```\
-- Q10. 월별 판매량 및 판매금액 : 연도, 월, 판매량, 판매금액  (안배웠지만 구글링 합시다. -> SQL도 ‘함수’ 란게 존재)
SELECT MID(OrderDate, 1, 4), MID(OrderDate, 6, 7), SUM(Quantity), SUM(Price) 
FROM (
  SELECT *
  FROM (
    SELECT *
    FROM Orders
    ) O
    JOIN OrderDetails Ord
    ON O.OrderID = Ord.OrderID
    ) OOrd
    JOIN Products P
    ON OOrd.ProductID = P.ProductID
GROUP BY MID(OrderDate, 6, 7)
```

