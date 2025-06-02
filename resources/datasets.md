# 공개 데이터셋 목록 및 설명

이 문서는 AIoT 연구에 활용할 수 있는 공개 데이터셋을 정리한 것입니다. 각 데이터셋의 특성, 접근 방법, 활용 사례를 포함하고 있습니다.

## 목차
1. [시계열 센서 데이터셋](#1-시계열-센서-데이터셋)
2. [활동 인식 데이터셋](#2-활동-인식-데이터셋)
3. [스마트 홈/빌딩 데이터셋](#3-스마트-홈빌딩-데이터셋)
4. [산업용 IoT 데이터셋](#4-산업용-iot-데이터셋)
5. [에너지 및 환경 데이터셋](#5-에너지-및-환경-데이터셋)
6. [교통 및 모빌리티 데이터셋](#6-교통-및-모빌리티-데이터셋)
7. [데이터셋 활용 팁](#7-데이터셋-활용-팁)

## 1. 시계열 센서 데이터셋

### 1.1 UCI Machine Learning Repository 시계열 데이터셋

- **설명**: 다양한 도메인의 시계열 데이터셋 모음
- **주요 데이터셋**:
  * **Air Quality Dataset**: 이탈리아 도시의 대기 오염 측정 데이터
  * **Appliances Energy Prediction**: 스마트 홈의 가전제품 에너지 소비 데이터
  * **PEMS-SF**: 고속도로 점유율 및 교통량 데이터
- **접근 방법**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets.php)에서 직접 다운로드
- **활용 사례**: 시계열 예측, 이상 탐지, 패턴 인식

### 1.2 Numenta Anomaly Benchmark (NAB)

- **설명**: 실시간 이상 탐지를 위한 시계열 데이터셋
- **데이터 특성**:
  * 58개의 실제 및 인공 시계열 데이터
  * 레이블된 이상(anomaly) 포함
  * 다양한 도메인(IT, 교통, 센서 등)
- **접근 방법**: [GitHub Repository](https://github.com/numenta/NAB)
- **활용 사례**: 이상 탐지 알고리즘 벤치마킹, 실시간 모니터링 시스템

### 1.3 NASA Bearing Dataset

- **설명**: 베어링 고장 예측을 위한 진동 센서 데이터
- **데이터 특성**:
  * 4개 베어링의 수명 주기 진동 데이터
  * 1초당 20,480개 샘플의 고주파 데이터
  * 고장 시점까지의 모든 데이터 포함
- **접근 방법**: [NASA Prognostics Center](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- **활용 사례**: 예측 유지보수, 고장 예측, 잔여 수명 추정

## 2. 활동 인식 데이터셋

### 2.1 UCI Human Activity Recognition (HAR)

- **설명**: 스마트폰 센서를 이용한 인간 활동 인식 데이터셋
- **데이터 특성**:
  * 30명의 참가자, 6가지 활동(걷기, 앉기 등)
  * 가속도계 및 자이로스코프 데이터
  * 2.56초 윈도우, 50% 오버랩
- **접근 방법**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **활용 사례**: 활동 인식, 웨어러블 컴퓨팅, 행동 패턴 분석

### 2.2 OPPORTUNITY Activity Recognition

- **설명**: 다양한 센서로부터 수집된 일상 활동 데이터셋
- **데이터 특성**:
  * 4명의 참가자, 다양한 일상 활동
  * 72개의 환경 및 신체 부착 센서
  * 17개의 활동 클래스와 미세한 제스처 레이블
- **접근 방법**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition)
- **활용 사례**: 복합 활동 인식, 컨텍스트 인식, 센서 퓨전

### 2.3 WISDM (Wireless Sensor Data Mining)

- **설명**: 스마트폰 및 스마트워치 센서 기반 활동 인식 데이터셋
- **데이터 특성**:
  * 다양한 일상 활동(걷기, 조깅, 계단 오르기 등)
  * 가속도계 데이터 (3축)
  * 다양한 사용자 그룹
- **접근 방법**: [WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php)
- **활용 사례**: 웨어러블 기기 기반 활동 인식, 건강 모니터링

## 3. 스마트 홈/빌딩 데이터셋

### 3.1 CASAS Smart Home Dataset

- **설명**: 스마트 홈 환경에서 수집된 일상 활동 데이터
- **데이터 특성**:
  * 여러 실제 주거 환경에서 수집
  * 동작, 온도, 문 센서 등 다양한 센서 데이터
  * 레이블된 활동 정보
- **접근 방법**: [CASAS 웹사이트](http://casas.wsu.edu/datasets/)
- **활용 사례**: 활동 인식, 스마트 홈 자동화, 행동 패턴 예측

### 3.2 UK-DALE (UK Domestic Appliance-Level Electricity)

- **설명**: 가정용 전력 소비 데이터셋
- **데이터 특성**:
  * 5개 가정의 전력 소비 데이터
  * 전체 및 기기별 전력 소비량
  * 최대 3년간의 장기 데이터
- **접근 방법**: [UK-DALE Dataset](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017)
- **활용 사례**: 전력 소비 예측, NILM(Non-Intrusive Load Monitoring), 에너지 관리

### 3.3 BuildSys 2021 Data Challenge

- **설명**: 상업용 건물의 HVAC 시스템 운영 데이터
- **데이터 특성**:
  * 실제 상업 건물의 HVAC 센서 데이터
  * 온도, 습도, CO2, 점유율 등
  * 에너지 소비 및 환경 데이터
- **접근 방법**: [BuildSys Data Challenge](https://buildsys.acm.org/2021/challenge/)
- **활용 사례**: 에너지 효율 최적화, HVAC 제어, 실내 환경 품질 예측

## 4. 산업용 IoT 데이터셋

### 4.1 PHM Data Challenge 데이터셋

- **설명**: Prognostics and Health Management 학회 제공 장비 상태 모니터링 데이터
- **주요 데이터셋**:
  * **PHM 2016**: 산업용 냉각 시스템 고장 데이터
  * **PHM 2018**: CNC 기계 공구 마모 데이터
  * **PHM 2019**: 유압 시스템 고장 데이터
- **접근 방법**: [PHM Society Data Challenge](https://www.phmsociety.org/competition/challenges)
- **활용 사례**: 예측 유지보수, 고장 진단, 잔여 수명 예측

### 4.2 MIMII Dataset (Malfunctioning Industrial Machine Investigation and Inspection)

- **설명**: 산업용 기계의 소리 데이터셋
- **데이터 특성**:
  * 4종류의 산업 기계(팬, 펌프, 밸브, 슬라이드 레일)
  * 정상 및 비정상 작동 소리
  * 다양한 SNR(신호 대 잡음비) 수준
- **접근 방법**: [Zenodo Repository](https://zenodo.org/record/3384388)
- **활용 사례**: 소리 기반 이상 탐지, 산업 기계 상태 모니터링

### 4.3 Tennessee Eastman Process Dataset

- **설명**: 화학 공정 시뮬레이션 데이터셋
- **데이터 특성**:
  * 52개의 변수(41개 측정, 11개 조작 변수)
  * 정상 작동 및 20가지 고장 시나리오
  * 화학 공정 전문 지식 포함
- **접근 방법**: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1)
- **활용 사례**: 공정 모니터링, 고장 탐지 및 진단, 제어 시스템 개발

## 5. 에너지 및 환경 데이터셋

### 5.1 Solar and Wind Power Forecasting

- **설명**: 미국 국립 재생 에너지 연구소(NREL)의 태양광 및 풍력 발전 데이터
- **데이터 특성**:
  * 실제 발전소의 출력 데이터
  * 기상 조건 및 발전량
  * 다양한 지역 및 시간대
- **접근 방법**: [NREL Data Catalog](https://data.nrel.gov/)
- **활용 사례**: 재생 에너지 예측, 전력 그리드 관리, 에너지 최적화

### 5.2 Intel Berkeley Research Lab Data

- **설명**: 연구실 환경에서 수집된 센서 데이터
- **데이터 특성**:
  * 54개 센서의 온도, 습도, 조도, 전압 데이터
  * 30초 간격으로 수집된 230만 개 이상의 측정값
  * 센서 위치 정보 포함
- **접근 방법**: [Intel Lab Data](http://db.csail.mit.edu/labdata/labdata.html)
- **활용 사례**: 센서 네트워크 연구, 실내 환경 모니터링, 센서 보정

### 5.3 Air Quality Dataset

- **설명**: 다양한 도시의 대기 질 측정 데이터
- **데이터 특성**:
  * PM2.5, PM10, NO2, SO2, O3, CO 등 오염물질 농도
  * 시간별 측정 데이터
  * 기상 조건(온도, 습도, 풍속 등) 포함
- **접근 방법**: [OpenAQ](https://openaq.org/) 또는 [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)
- **활용 사례**: 대기 오염 예측, 환경 모니터링, 건강 영향 분석

## 6. 교통 및 모빌리티 데이터셋

### 6.1 T-Drive Taxi Trajectories

- **설명**: 베이징 택시의 GPS 궤적 데이터
- **데이터 특성**:
  * 10,357대 택시의 1주일간 GPS 궤적
  * 총 1,500만 개 이상의 데이터 포인트
  * 3~5분 간격의 위치 정보
- **접근 방법**: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)
- **활용 사례**: 교통 패턴 분석, 이동 경로 예측, 스마트 시티 계획

### 6.2 CRAWDAD Wireless Network Data

- **설명**: 무선 네트워크 및 모빌리티 데이터 아카이브
- **데이터 특성**:
  * WiFi, Bluetooth, 셀룰러 네트워크 데이터
  * 다양한 모빌리티 시나리오
  * 실제 환경에서 수집된 데이터
- **접근 방법**: [CRAWDAD Website](https://crawdad.org/)
- **활용 사례**: 네트워크 성능 분석, 모빌리티 예측, 위치 기반 서비스

### 6.3 City Pulse EU FP7 프로젝트 데이터셋

- **설명**: 스마트 시티 관련 IoT 데이터셋
- **데이터 특성**:
  * 교통 흐름, 주차, 대기 오염 등 도시 데이터
  * 소셜 미디어 데이터 통합
  * 실시간 및 이력 데이터
- **접근 방법**: [City Pulse Dataset](http://iot.ee.surrey.ac.uk:8080/datasets.html)
- **활용 사례**: 도시 모니터링, 교통 예측, 시민 서비스 개발

## 7. 데이터셋 활용 팁

### 7.1 데이터 전처리 권장사항

- **결측치 처리**:
  * 시계열 데이터의 경우 보간법(interpolation) 사용 고려
  * 센서 데이터의 특성에 맞는 결측치 처리 방법 선택
  * 결측 패턴 분석을 통한 인사이트 도출

- **정규화 및 표준화**:
  * 센서별 스케일 차이 고려
  * 전체 또는 윈도우 기반 정규화 선택
  * 도메인 지식을 활용한 특성별 정규화 방법 결정

### 7.2 데이터셋 평가 기준

- **데이터 품질**:
  * 결측치 비율 및 분포
  * 노이즈 수준 및 이상치 존재 여부
  * 데이터 수집 방법 및 센서 사양

- **적합성 검토**:
  * 연구 목적에 맞는 센서 종류 및 특성
  * 충분한 데이터 양과 다양성
  * 레이블 정확도 및 신뢰성

- **법적/윤리적 고려사항**:
  * 데이터 사용 라이센스 확인
  * 개인정보 보호 준수
  * 적절한 인용 및 출처 명시

---

**참고**: 이 목록은 계속 업데이트됩니다. 추가하고 싶은 데이터셋이 있다면 연구실 GitHub 저장소에 기여해주세요.

**마지막 업데이트**: 2025-05-30