# 1. Pandas 데이터 그룹처리

**기준 데이터 프레임**

```python
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings','Kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}

df=pd.DataFrame(ipl_data)
df
```



- sum()

  - ``df.groupby('Team')['Points'].sum()``

    Team을 기준으로 Points값을 합계한 Series를 반환

  - ``df.groupby(['Team','Year'])['Points'].sum()``

    기준의 값이 복수 개일 경우, 먼저 입력한 기준이 더 큰 범주를 가지게 된다.

  - ``c_df.sum(level=0)``

    1번째 기준을 기준으로 합계를 계산한다.

  - ``c_df.sum(level=1)``

    2번째 기준을 기준으로 합계를 계산한다.

- index

  - ``c_df.index``

    기준 인덱스의 값을 튜플 형태로 반환한다.

  - ``c_df['Devils':'Kings']``

    데이터를 불러올 기준의 범위를 지정할 수 있다.

- unstack()

  - ``c_df.unstack()``

    기존의 데이터를 표의 형태로 펼쳐서 볼 수 있다

- swaplevel()

  - ``c_df.swaplevel().sort_index()``

    기준을 변경하고 인덱스를 기준으로 정렬한다. 단, 데이터는 바뀌지 않는다.

- get_group()

  - ``dc_df.get_group("Riders")``

    지정한 그룹의 데이터만 가져온다

- agg()

  - 함수를 인자 값으로 받아서 그 연산을 수행한다.

  - ```python
    # agg(min): 각 그룹별 가장 작은 값을 가져온다
    dc_df.agg(min)
    
    # numpy 연산을 가져와서 사용할 수 있다
    # np.mean : 평균값 계산
    dc_df.agg(np.mean)
    
    # max 옵션 : 최대 값
    dc_df.agg(max)
    ```

- transform()

  - 함수를 지정해서 그 연산을 수행시킬 수 있다.

  - ```python
    f=lambda x:(x-x.mean())/x.std()
    dc_df.transform(f)
    ```

- filter()

  - 함수를 지정해서 그 조건에 부합하는 데이터를 추출한다.

  - ```python
    # 길이가 3이상인 데이터만 추출
    dc_df.filter(lambda x:len(x)>=3)
    
    # Points의 최대 값이 800이상인 데이터만 추출
    dc_df.filter(lambda x:x['Points'].max() >= 800)
    ```

- merge()

  - 두 데이터프레임을 병합한다.

  - ```python
    # how="inner": 삽입 - 연관성이 없으면 배제한다
    # 즉, 데이터가 있는 것만 결합
    # on : 기준
    pd.merge(left=ck,right=ck2,how="inner",on='Team')
    
    # 오른쪽에 놓은 프레임 기준으로 병합한다.
    pd.merge(ck,ck2,on="Team",how="right")
    
    # 왼쪽에 놓은 프레임 기준
    pd.merge(ck,ck2,on="Team",how="left")
    
    # outer : 두 프레임의 내용을 모두 취합해서 연결(합집합)
    pd.merge(ck,ck2,on="Team",how="outer")
    ```

- concat()

  - 두 프레임을 axis(축)을 기준으로 연결한다

  - ```python
    # axis는 축으로 차원이다. 0 : n차원, 1 : n-1차원 ...
    pd.concat([t1,t2],axis=1).reset_index(drop=True)
    ```

- append()

  - 데이터프레임을 뒤에 이어 붙인다

  - 기준 프레임에 따라서 순서가 달라지며, 순차적인 결합을 진행할 때 사용한다

  - ```python
    # t1을 기준으로 t2를 연결
    end_df=t1.append(t2).reset_index(drop=True)
    
    # t2을 기준으로 t2를 연결
    end_df=t2.append(t1).reset_index(drop=True)
    ```