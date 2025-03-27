# IoT 센서 데이터 처리 튜토리얼

이 디렉토리에는 AIoT 연구에 필수적인 IoT 센서 데이터 수집, 전처리, 분석 및 시각화 방법에 대한 튜토리얼이 포함되어 있습니다.

## 학습 목표

1. 다양한 IoT 센서 유형 및 특성 이해
2. 센서 데이터 수집 및 저장 방법 습득
3. 시계열 데이터 전처리 및 특징 추출 기법 학습
4. 센서 데이터 분석 및 시각화 방법 익히기
5. AIoT 시스템 구축을 위한 기반 지식 확보

## 튜토리얼 구성

### 1. IoT 센서 기초 (sensor_basics)

- `01_sensor_types_characteristics.ipynb`: 센서 유형 및 특성 소개
- `02_data_formats_protocols.ipynb`: 센서 데이터 형식 및 통신 프로토콜
- `03_sampling_frequency.ipynb`: 샘플링 주파수 및 데이터 품질
- `04_sensor_calibration.ipynb`: 센서 보정 방법

### 2. 데이터 수집 (data_collection)

- `01_mqtt_basics.ipynb`: MQTT 프로토콜을 이용한 데이터 수집
- `02_rest_api_integration.ipynb`: REST API를 활용한 센서 데이터 수집
- `03_database_storage.ipynb`: 시계열 데이터베이스 저장 방법
- `04_data_streaming.ipynb`: 실시간 데이터 스트리밍 처리

### 3. 데이터 전처리 (preprocessing)

- `01_noise_filtering.ipynb`: 노이즈 필터링 및 신호 처리
- `02_missing_data_handling.ipynb`: 결측치 처리 기법
- `03_outlier_detection.ipynb`: 이상치 탐지 및 처리
- `04_feature_extraction.ipynb`: 시계열 데이터 특징 추출

### 4. 데이터 분석 및 시각화 (analysis_visualization)

- `01_time_series_visualization.ipynb`: 시계열 데이터 시각화 기법
- `02_statistical_analysis.ipynb`: 통계적 분석 방법
- `03_correlation_analysis.ipynb`: 다중 센서 데이터 상관관계 분석
- `04_interactive_dashboards.ipynb`: 인터랙티브 대시보드 구축

### 5. AIoT 응용 사례 (aiot_applications)

- `01_environmental_monitoring.ipynb`: 환경 모니터링 시스템
- `02_human_activity_recognition.ipynb`: 웨어러블 센서를 이용한 활동 인식
- `03_predictive_maintenance.ipynb`: 산업 장비 예측 유지보수
- `04_smart_energy_management.ipynb`: 스마트 에너지 관리 시스템

## 사전 요구사항

- Python 기초 프로그래밍 지식
- NumPy, Pandas 기본 사용법 이해
- 간단한 통계 개념 이해
- Matplotlib, Seaborn 등 시각화 라이브러리 기초 지식

## 환경 설정

튜토리얼을 진행하기 위한 환경 설정 방법입니다:

```bash
# 가상 환경 생성 및 활성화
conda create -n iot-sensor python=3.8
conda activate iot-sensor

# 기본 패키지 설치
pip install numpy pandas matplotlib seaborn jupyterlab

# 시계열 데이터 처리 패키지
pip install statsmodels scipy tslearn

# IoT 관련 패키지
pip install paho-mqtt requests influxdb

# 대화형 시각화 패키지
pip install plotly dash bokeh
```

## 실습에 사용되는 센서 데이터셋

튜토리얼에서는 다음과 같은 공개 데이터셋을 활용합니다:

1. **UCI HAR Dataset**: 스마트폰 센서를 이용한 인간 활동 인식 데이터
2. **Intel Berkeley Lab Dataset**: 실내 환경 모니터링 센서 데이터
3. **NASA Bearing Dataset**: 베어링 고장 예측을 위한 진동 센서 데이터
4. **NREL Solar and Wind Dataset**: 태양광 및 풍력 발전소 센서 데이터

데이터셋은 튜토리얼 진행 시 자동으로 다운로드되거나, 별도 안내에 따라 수동으로 다운로드하여 사용합니다.

## 시계열 데이터 처리 핵심 개념

### 1. 시계열 데이터 특성

- **시간적 의존성**: 데이터 포인트 간의 시간적 순서와 의존성
- **계절성**: 일정 주기로 반복되는 패턴
- **추세**: 장기적인 증가 또는 감소 경향
- **정상성(Stationarity)**: 시간에 따른 통계적 속성의 일관성

### 2. 주요 전처리 기법

- **리샘플링**: 다른 시간 간격으로 데이터 변환
- **이동 평균**: 노이즈 감소 및 추세 파악
- **차분**: 추세 제거 및 정상성 확보
- **로그 변환**: 분산 안정화

### 3. 특징 추출 방법

- **통계적 특징**: 평균, 표준편차, 분위수, 첨도, 왜도 등
- **주파수 도메인 특징**: FFT, 파워 스펙트럼, 스펙트럴 엔트로피
- **시간-주파수 특징**: 웨이블릿 변환
- **정보 이론 특징**: 엔트로피, 상호 정보량

## 실습 프로젝트 아이디어

튜토리얼 학습 후 시도해볼 수 있는 프로젝트 아이디어:

1. **개인 활동 모니터링 시스템**: 스마트폰/웨어러블 센서를 활용한 일상 활동 패턴 분석
2. **스마트홈 환경 모니터링**: 온도, 습도, CO2 등 실내 환경 요소 모니터링 및 예측
3. **교통 흐름 분석**: 도로 센서 데이터를 활용한 교통 패턴 분석 및 혼잡도 예측
4. **설비 상태 모니터링**: 진동, 온도 센서를 이용한 장비 이상 탐지 시스템

## IoT 데이터 처리 파이프라인

효과적인 IoT 데이터 처리 파이프라인 구축 단계:

1. **데이터 수집**: 센서로부터 원시 데이터 수집
2. **데이터 전송**: 에지 디바이스에서 게이트웨이 또는 클라우드로 전송
3. **데이터 저장**: 적절한 데이터베이스에 저장 (시계열DB, 관계형DB 등)
4. **데이터 전처리**: 노이즈 제거, 결측치 처리, 이상치 탐지
5. **특징 추출**: 의미 있는 특징 도출
6. **모델링/분석**: ML/DL 모델 적용 또는 통계적 분석
7. **시각화/알림**: 결과 시각화 및 필요시 알림 생성

## 자주 사용되는 Python 라이브러리

### 데이터 처리 및 분석
- **Pandas**: 시계열 데이터 조작 및 분석
- **NumPy**: 수치 계산 및 배열 처리
- **SciPy**: 과학적 계산 및 신호 처리
- **Statsmodels**: 통계 모델링 및 검정

### 시계열 특화 라이브러리
- **Prophet**: 시계열 예측
- **tslearn**: 시계열 머신러닝
- **tsfresh**: 자동화된 시계열 특징 추출
- **pyts**: 시계열 분류 및 변환

### IoT 통신 및 데이터베이스
- **paho-mqtt**: MQTT 클라이언트
- **requests**: HTTP/REST 통신
- **influxdb-client**: InfluxDB 연동
- **pymongo**: MongoDB 연동

### 시각화
- **Matplotlib**: 기본 그래프 작성
- **Seaborn**: 통계적 데이터 시각화
- **Plotly**: 인터랙티브 시각화
- **Dash**: 웹 대시보드 구축

## 참고 자료

- [Time Series Analysis with Python Cookbook](https://www.packtpub.com/product/time-series-analysis-with-python-cookbook/9781801075541)
- [Practical Time Series Analysis](https://www.oreilly.com/library/view/practical-time-series/9781492041641/)
- [IoT and Edge Computing for Architects](https://www.packtpub.com/product/iot-and-edge-computing-for-architects-second-edition/9781839214806)
- [Coursera - IoT & Sensor Data Analytics](https://www.coursera.org/specializations/sensor-data-analytics)

## 도움 받기

튜토리얼 진행 중 어려움이 있을 경우:
1. 연구실 Slack 채널 #iot-sensors에 질문
2. 지도교수 또는 선배 연구자에게 문의
3. 주간 IoT 스터디 그룹 미팅 참여

---

## IoT 센서 데이터 처리 체크리스트

### 데이터 수집 단계
- [ ] 센서 유형 및 특성 파악
- [ ] 적절한 샘플링 주파수 설정
- [ ] 데이터 수집 프로토콜 선택
- [ ] 데이터 저장 형식 결정

### 데이터 전처리 단계
- [ ] 결측치 확인 및 처리
- [ ] 노이즈 필터링
- [ ] 이상치 탐지 및 처리
- [ ] 데이터 정규화/표준화

### 특징 추출 단계
- [ ] 시간 도메인 특징 추출
- [ ] 주파수 도메인 특징 추출
- [ ] 통계적 특징 계산
- [ ] 특징 선택 및 차원 축소

### 분석 및 모델링 단계
- [ ] 탐색적 데이터 분석 수행
- [ ] 적절한 모델/알고리즘 선택
- [ ] 모델 학습 및 검증
- [ ] 모델 성능 평가

### 시각화 및 결과 해석 단계
- [ ] 핵심 지표 시각화
- [ ] 패턴 및 추세 시각화
- [ ] 인터랙티브 대시보드 구축
- [ ] 결과 해석 및 인사이트 도출