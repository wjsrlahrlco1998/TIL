# Django MVC



## 1. MVC란?



1. Django에서의 MVC -> MTV

   - M(Model) : DB와의 커넥션을 담당하는 영역
   - V(View) : 결과물을 생성하는 영역
   - C(Controller) : Model과 View를 핸들링하고, 비즈니스 로직을 처리하는 영역
     - Django에서는 ``views.py``가 담당.

2. Django Application 생성하기

   1. ``python manage.py startapp blog``

      - Django는 하나의 프로젝트에 대한 다양한 app을 구성하여 개발한다. 이는 app하나를 모듈화한 것이다.
      - 각 모듈 살펴보기
        - ``admin.py`` : admin 등록
        - ``apps.py`` : Application Config
        - ``models.py`` : DB관리
        - ``tests.py`` : 테스트 관련 파일
        - ``views.py`` : 사용자에게 보여주기 위한 파일

   2. views에 대한 url 연결

      - ``blog/urls.py`` 파일 생성

        ```python
        from django.urls import path
        from . import views
        
        urlpatterns = [
            path('', views.index, name='index'),
            path('list/', views.blog_list, name='blog_list'),
            path('list/<int:blog_id>', views.blog_detail, name='blog_detail')
        ]
        ```

      - ``myblog/settings.py`` 코드 추가

        ```python
        INSTALLED_APPS = [
            'django.contrib.admin',
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'django.contrib.sessions',
            'django.contrib.messages',
            'django.contrib.staticfiles',
            'blog',  # Project에 어떤 app을 사용할지 알려줌
        ```

      - ``myblog/urls.py`` 코드 추가

        ```python
        from django.contrib import admin
        from django.urls import path, include
        
        urlpatterns = [
            path('admin/', admin.site.urls),
            path('', include('blog.urls')),
        ]
        ```

      3. Rendering

         - ``veiws.py``

           ```python
           def index(request):
               print(request)
               print(request.headers)
               return render(request, 'index.html', {'name': '박지성', 'profile': profile_list})
           ```

         - ``index.html``

           ```php+HTML
           {% extends "base.html" %}
           {% block main %}
             <div class="container">
               <h1 class="mt-5">홈화면</h1>
               <p class="lead">Pin a footer to the bottom of the viewport in desktop browsers with this custom HTML and CSS. A fixed navbar has been added with <code class="small">padding-top: 60px;</code> on the <code class="small">main &gt; .container</code>.</p>
               <p>Back to <a href="/docs/5.2/examples/sticky-footer/">the default sticky footer</a> minus the navbar.</p>
             </div>
           {% endblock %}
           ```

           

         