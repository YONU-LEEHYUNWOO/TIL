# precommit

## precommit이란?

precommit은 commit전 필수로 점검해야 할 사항을 자동 수행하도록 도와주는 도구 

### how to ?

``` 
#설치 
$ pip install pre-commit 

#설치확인
$ pre-commit -v

#설정 
$ touch.pre-commit-config.yaml
$ pre-commit sample-comfig > .pre-commit-config.yaml

# 동작확인
$ pre-commit run

# 실제 사용
$ git add .
$ pre-commit run
(후 테스트에 실패하여 수정된 사항 다시 add)
$ git add .
$ git commit

# 자동 실행을 원한다면
$ pre-commit install
(이후 commit 명령 수행때 마다 자동실행함)
$ git commit
