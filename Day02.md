# Git 특강 2일차 정리

## 1. gitignore

#### 정의

- 버전관리 대상으로 지정하지 않는 파일의 내용을 담은 파일

#### 사용법

- **.gitignore**라는 파일을 생성한 후 그 안에 버전관리 대상으로 지정하지 않을 파일의 이름을 입력한다.

#### 사용이유

- 암호의 key 값, 개인정보 등의 파일을 보호하기 위해 보안의 측면에서 사용
- 관리의 필요성이 없는 가비지 파일들을 지정하기 위해서 사용

#### 쉽게 작성하는 방법

- 웹사이트 이용 
  - ignore 파일을 자동으로 생성하여 그 내용을 복사 붙여넣기 하여 사용하면 된다.
  - [https://www.toptal.com/developers/gitignore](https://www.toptal.com/developers/gitignore)

- gitignore 저장소 이용
  - 이미 언어별, 프로젝트별로 만들어진 ignore 파일을 이용한다.
  - [https://github.com/github/gitignore](https://github.com/github/gitignore)

#### 주의 사항

- 이미 한번 버전관리에 들어간 파일은 적용되지 않는다.

## 2. GitHub로부터 받아오기

#### 명령어

- ``git clone {URL}``

  - 정의 

    해당 URL의 commit을 그대로 복제한다.

  - 특징

    - 원격저장소에 해당하는 폴더를 생성
    - commit을 복제
    - init, remote를 모두 자동으로 해줌

  - 다른 형태

    - ``git clone {URL} {폴더이름}``

      지정 폴더의 이름으로 폴더 생성

- ``git pull {별명} master``

  - 정의

    {별명}으로 등록된 URL의 내용을 받아와 내 로컬 저장소의 내용을 업데이트한다.

  - **git pull의 비정상적인 흐름**

    원격저장소의 내용이 업데이트 되었는데, pull 받지 않고 나의 업데이트 내용을 push하는 경우 - 충돌발생

    ***해결법***

    1. ``pull origin master``을 실행
    2. 최신버전 선택
    3. 나의 업데이트 내용을 추가해서 새로운 버전을 다시 add->commit->push

## 3. Branch

#### 정의

- master(root)의 현 시점의 commit을 그대로 가져온다. 그러나 master와는 다른 공간이므로 Merge 전에는 master에 영향을 주지 못하고 master의 업데이트 내용을 반영하지 못한다.

#### 특징

- 병렬적 협업을 가능하게 한다.

#### 명령어

- ``git branch``: 가지고 있는 branch의 목록을 알려줌
- ``git branch {별명}``: 해당 별명으로 branch 생성
- ``git switch {별명}``: 해당 별명의 branch로 이동
- ``git branch -d {별명}``: 해당 별명의 branch를 제거
- ``git merge {branch 이름}``: 해당 이름의 브랜치와 합친다.

#### 브랜치 방식

1. Fast - Forward 

   길을 따라서 쭉 합치는 방식

2. Auto - Merging 

   갈라진 길을 따라서 쭉 합치는 것

3. conflict

   합칠 branch와 합쳐지는 branch가 모두 수정을 했을 경우 충돌이 발생한다. 해결방법은 pull의 rejected 상황과 같다.

#### 주의사항

- commit을 하지않으면 branch 이동이 불가능하다.
- 현재 branch의 위치에 유의
- 수명이 끝난 branch는 삭제하는 것이 좋다.
- Merge는 기능을 추가해야하는 branch에서 수행한다.

#### branch의 사용예시

- master가 현재 시스템(사용중)인데, 만약 문제가 발생하면 현재 master에서 branch(hot-fix)를 생성한 후, 거기서 해결한 후 merge해서 master을 업데이트한다.

## 4. Git Work-Flow

#### 정의

- GitHub를 통한 협업방식의 흐름

#### Pull Request(PR)

- Pull을 해도되는지 요청하는 것

#### Work-Flow

1. 소유권이 없는 원격저장소를 fork하여 복제공간을 생성

2. 복제공간을 ``git clone {URL}``을 통해서 로컬저장소 생성

3. ``git remote add upstream {원본URL}``을 통해서 원본저장소와 동기화

4. ``git branch {별명}``을 통해서 기능을 추가할 branch 생성

5. 생성한 branch를 add -> commit -> push, **여기서 push 진행시 기능을 추가한 branch를 push 해야한다.**

   ex) ``git push origin {생성한branch}`` 

   (또한 push는 복제공간에 하는 것이다. 원본 공간에는 push가 불가능하다.)

6. 복제공간에서 원본공간에 Pull Request를 보낸다.(단, 병합할 branch에 보내야한다.)

7. 원본공간에서 PR을 허락하면 원본에 merge되고 복제공간에서 merge된 branch는 디폴트로 자동삭제된다.

8. 이후 원본을 다시 pull 받고, 기능추가, 복제공간에 push, PR이 반복된다.

## 5. 원칙

- 프로젝트 진행 시, README.md와 .gitignore을 먼저 생성
