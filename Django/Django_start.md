# Django 시작



## 1. Django란?

- Deadline이 있는 완벽주의자를 위한 웹프레임워크
  - 즉, 빠르게 웹을 만들 수 있는 프레임워크
  - 통신, DB, 쿠기 및 세션, HTTP Header, Body, Routing, Template Engine 등을 구현해 놓음



## 2. Django 개발환경 설정



1. 가상환경 생성 및 django 설치

```bash
conda create -n py38_django python=3.8
conda activate py38_django
conda install django==3.2.5
```

- Anaconda 사용시 호환성을 위해 django3 버전을 설치한다.



## 3. Django 시작하기



1. django 프로젝트 시작하기

   - django-admin tool set 사용
   - ``django-admin startproject <프로젝트 이름>``

   ```shell
   django-admin startproject myblog
   cd myblog
   code . # vscode로 열기
   ```

2. django 프로젝트 디렉토리 살펴보기

   - manage.py
     - 해당 django project를 편하게 조작할 수 있는 CLI 유틸리티
   - \<project>/settings.py
     - 해당 project의 설정파일
   - \<project>/urls.py
     - 해당 project의 url 구성
   - \<project>/wsgi.py, \<project>/asgi.py
     - Web(Asyncronous)SeverGatewayInterface의 약자로 deploy(배포)를 위한 파일

3. django 프로젝트 시작하기

   ```powershell
   python manage.py runserver
   ```

   - 표시되는 주소로 접속하면 서버가 시작됨을 알 수 있다.

4. Database 연결하기

   - https://docs.djangoproject.com/en/4.0/ref/settings/#databases

   - 위의 주소를 참고하여 Database를 연동하자.

   - 연동 후

     ```shell
     python manage.py migrate # DB에 내용 반영
     python manage.py createsuperuser # 관리자 계정 생성
     ```

   - localhost:8000/admin 으로 접속하면 관리자 화면이 표시된다.