# Git 특강 3일차 정리

## 1. Pull Request 심화

1. 원격 저장소의 관리자인 경우

   ```
   1. PR 수락 
   2. merge
   3. conflict 발생
   4. 해결
   5. merge and commit
   ```

2. 원격 저장소의 소유권이 없는 경우

   ```
   1. Fork
   2. clone 생성
   3. 기능을 추가할 branch 생성
   4. 기능 추가
   5. add - commit - push
   6. PR 송신
   7. 원본 pull
   8. 사용한 branch 삭제
   ```

3. 자기자신에게 PR을 보내는 경우

   ```
   1. branch 생성
   2. 기능 추가
   3. git push origin {branch name}
   4. 나에게 온 PR 수락
   5. branch 삭제
   ```
   

**GitHub에서 PR 충돌 해결은 수동으로 수정한다.**



## 2. 복구

- **git status**의 동작

  git status는 Working Directory, Staging Area, Commit의 공간을 W.D <-> S.A, S.A <-> Commit 로 비교해서 상태의 차이를 보여준다.

  1. W.D와 S.A 비교

     W.D의 파일이 S.A에도 존재하는지 비교

  2. S.A와 Commit 비교

     S.A의 파일이 Commit에도 존재하는지 비교

  3. '1'과 '2'가 모두 충족되면 Nothing Commit

- 삭제의 동작

  ex) add까지 한 파일을 삭제하는 경우

  : W.D에서만 지워지는 것이므로 S.A에는 남아있다.

  *W.D에서 삭제하고 add하면 삭제가 반영되어서 S.A에서도 없어진다*

#### 1) 버전 되돌리기

- 명령어

  ``git reset {옵션} {해시값:commit ID}``

  해당 해시 값을 가진 commit으로 옵션에 따라서 되돌린다.

  <옵션>

  1. ``--sort`` : 앞의 버전을 Commit에서만 제거한다. (W.D와 S.A에 존대한다)
  2. ``--mixed`` : 앞의 버전을 S.A와 Commit에서 제거한다.
  3. ``--hard`` : 앞의버전을 모든 공간에서 지운다.

#### 2) 버전 돌아가기

- 명령어

  ``git reflog`` : 지금까지의 모든 커밋이 기록되어있어서 돌아가고 싶은 버전의 해시 값을 얻는다.

  ``git reset --hard {해시값}`` : 해시 값에 해당하는 커밋으로 돌아간다.

#### 3) revert

-  정의 : 커밋을 취소한 상태 자체를 커밋으로 올리는 기능으로 reset을 사용할 때 다른 협업자와 커밋의 상태가 꼬이는 것을 방지하기위해 개선한 형태이다.

- 명령어

  ``git revert {해시 값}`` : 해시 값에 해당하는 커밋을 취소하고 그 상태를 커밋한다.

  (``git revert {해시 1} {해시 2}``의 형태로 여러 개를 지정할 수 있다)

---

**추가 지식**

- ``git push -u origin master`` 

  한번 실행하면 다음에는 git push로만 실행해도 origin master을 자동으로 인식한다.
