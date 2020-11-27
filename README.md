# 설치

- 파이썬 3.7.2 기준 (아나콘다 환경 X)

## 파이썬 설치

- https://www.python.org/downloads/release/python-372/

## pip 설치

- (recordapp 경로에서)

```
python get-pip.py
```

## 가상환경 설치

- (recordapp 경로에서)

```
python -m venv venv
```

```
cd venv/Scripts
```

```
activate.bat
```

## 패키지 설치

- (recordapp 경로에서)

```
pip install -r requirements-new.txt
```

- (recordapp 경로에서)

```
pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
```

# 실행

- (recordapp 경로에서)

```
python Main.py
```

# 유의사항

- 아나콘다 환경에서 파이썬 환경을 구축한 경우 작동되지 않습니다. 아나콘다를 지우고 직접 파이썬을 설치하고, pip로 패키지 관리를 해야합니다.

- 저장 폴더는 recordapp 폴더 안에서만 작동합니다. 다른 경로 지정시 에러 발생

- 한번 녹화하고 재녹화시 프로그램을 다시 시작해야 합니다.
