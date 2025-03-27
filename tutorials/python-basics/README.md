# Python 기초 튜토리얼

이 디렉토리에는 AIoT 연구에 필요한 Python 기초 기술을 학습할 수 있는 튜토리얼이 포함되어 있습니다.

## 학습 목표

1. Python 기본 문법 및 자료구조 이해
2. 데이터 처리 및 분석 기법 습득
3. 파일 입출력 및 데이터 포맷 다루기
4. 모듈화 및 객체지향 프로그래밍 이해

## 튜토리얼 구성

### 1. Python 기초 (basics)

- `01_variables_and_types.ipynb`: 변수, 데이터 타입, 기본 연산
- `02_control_flow.ipynb`: 조건문, 반복문, 예외 처리
- `03_functions.ipynb`: 함수 정의, 매개변수, 반환값
- `04_data_structures.ipynb`: 리스트, 튜플, 딕셔너리, 집합

### 2. 데이터 처리 (data_processing)

- `01_numpy_basics.ipynb`: NumPy 배열 및 연산
- `02_pandas_basics.ipynb`: Pandas DataFrame 기초
- `03_data_cleaning.ipynb`: 결측치, 이상치 처리
- `04_data_transformation.ipynb`: 데이터 변환 및 정규화

### 3. 데이터 시각화 (visualization)

- `01_matplotlib_basics.ipynb`: 기본 그래프 그리기
- `02_advanced_plots.ipynb`: 다양한 시각화 기법
- `03_interactive_viz.ipynb`: 인터랙티브 시각화
- `04_visualization_best_practices.ipynb`: 효과적인 시각화 방법

### 4. 파일 및 데이터 포맷 (file_io)

- `01_file_operations.ipynb`: 텍스트 파일 읽기/쓰기
- `02_csv_json_handling.ipynb`: CSV, JSON 포맷 다루기
- `03_binary_data.ipynb`: 바이너리 데이터 처리
- `04_database_access.ipynb`: 데이터베이스 연결 및 쿼리

### 5. 고급 주제 (advanced)

- `01_object_oriented.ipynb`: 클래스 및 객체 지향 프로그래밍
- `02_functional_programming.ipynb`: 함수형 프로그래밍 기법
- `03_decorators_generators.ipynb`: 데코레이터 및 제너레이터
- `04_parallel_processing.ipynb`: 병렬 처리 및 성능 최적화

## 사전 요구사항

- Python 3.8 이상 설치
- Anaconda 또는 필요한 패키지(numpy, pandas, matplotlib) 설치
- Jupyter Notebook 또는 JupyterLab 설치

## 학습 방법

1. 각 노트북 파일을 순서대로 실행하며 학습
2. 코드 예제를 직접 수정하고 실행해보며 이해
3. 각 섹션 끝에 있는 연습 문제 풀이
4. 학습 후 미니 프로젝트를 통한 종합 실습

## 미니 프로젝트 예시

학습한 내용을 종합적으로 적용할 수 있는 미니 프로젝트 주제:

1. **센서 데이터 분석**: IoT 센서에서 수집된 시계열 데이터 전처리 및 시각화
2. **이상치 탐지**: 데이터셋에서 이상치를 탐지하고 시각화하는 스크립트 개발
3. **데이터 대시보드**: 특정 데이터셋에 대한 기본 통계 및 시각화 대시보드 구현
4. **로그 분석기**: 시스템 로그 파일을 분석하여 중요 정보 추출 및 요약

## 참고 자료

- [Python 공식 문서](https://docs.python.org/3/)
- [NumPy 튜토리얼](https://numpy.org/doc/stable/user/tutorials_index.html)
- [Pandas 튜토리얼](https://pandas.pydata.org/docs/getting_started/index.html)
- [Matplotlib 튜토리얼](https://matplotlib.org/stable/tutorials/index.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

## 도움이 필요하다면

튜토리얼 진행 중 질문이나 문제가 있다면 다음 방법으로 도움을 요청하세요:

1. 연구실 Slack 채널 #python-help에 질문 게시
2. 주간 미팅에서 문제 공유
3. 지도교수 또는 선배 연구자에게 문의

---

## 01. Python 기초 모듈 시작하기

Python 기초 모듈을 시작하려면:

1. Jupyter Notebook을 실행합니다:
   ```
   jupyter notebook
   ```

2. 브라우저에서 `01_variables_and_types.ipynb` 파일을 열어 첫 번째 튜토리얼을 시작하세요.

3. 각 코드 셀을 순서대로 실행하며 내용을 학습합니다.

4. 제공된 연습 문제를 풀어보며 이해도를 점검하세요.