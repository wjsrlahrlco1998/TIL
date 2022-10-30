# Nodejs - Python 연동

> Nodejs와 Python의 연동 및 상호간 데이터 교환에 대한 방법을 공부한다.



- [Child Process를 이용한 연동](https://github.com/wjsrlahrlco1998/TIL/blob/master/Integration_with_Nodejs/child_process.md)

  

## * 환경

> Nodejs - Python 과의 연동에서 필자는 딥러닝 환경 즉 Anaconda 환경의 패키지가 필요하기 때문에, Anaconda 환경에 nodejs를 설치하여 실행하였다.

1. Anaconda 환경에서 Nodejs 설치
   - ``conda activate 가상환경이름`` 으로 가상환경 접속
   - ``conda install nodejs - c conda-forge --repodata-fn=repodata.json `` 으로 nodejs 설치