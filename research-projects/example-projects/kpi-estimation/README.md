# KPI Estimation 프로젝트

## 프로젝트 소개

KPI Estimation 프로젝트는 5G 네트워크 환경에서 수집된 시계열 데이터를 활용하여 핵심 성과 지표(KPI, Key Performance Indicators)를 예측하는 연구로, 이 프로젝트는 AIoT Lab의 시계열 예측 기술을 5G 네트워크 성능 분석에 적용한 실제 응용 사례를 보여줌

## 목표

- 5G 네트워크 KPI 데이터에 최적화된 예측 모델 개발
- Dense 네트워크와 Transformer 기반 시계열 예측 모델의 성능 비교 분석
- 5G 네트워크 특성(채널, 컨텍스트, 셀 관련 메트릭)을 고려한 모델링 방법 탐구
- 실제 네트워크 환경에서 활용 가능한 KPI 예측 시스템 개발

## 데이터셋

- **데이터 소스**: https://github.com/uccmisl/5Gdataset
- **수집 패턴**: 
  - 이동성 패턴: static, car
  - 애플리케이션 패턴: video streaming, file download
- **포함 메트릭**: 
  - 채널 관련 메트릭 (Channel-related metrics)
  - 컨텍스트 관련 메트릭 (Context-related metrics)
  - 셀 관련 메트릭 (Cell-related metrics)
  - 처리량 정보 (Throughput information)
- **수집 도구**: G-NetTrack Pro (Android 네트워크 모니터링 애플리케이션)

## 방법론

1. **데이터 전처리 및 특성**
   - 5G 네트워크 메트릭 정규화
   - KPI: Standard normalization 적용
   - Input features: MinMax normalization 적용
   - 시계열 특성에 맞는 윈도우 기반 전처리

2. **모델링 접근법**
   - **Dense Network**: 기본 완전 연결 신경망 모델
   - **Transformer**: 시계열 데이터를 위한 어텐션 기반 모델
   - 두 모델의 성능 비교 및 분석

3. **모델 학습 및 평가**
   - 5G 네트워크 환경 특성을 고려한 학습/검증 분할
   - 평가 지표: RMSE, MAE 사용
   - 예측 결과 시각화 및 분석

## 주요 실험 결과

5G 네트워크 KPI 예측에서 두 모델의 성능 비교:

| 모델 | RMSE | MAE | 특징 |
|------|------|-----|------|
| Dense | [실험 후 업데이트] | [실험 후 업데이트] | 빠른 학습, 간단한 구조 |
| Transformer | [실험 후 업데이트] | [실험 후 업데이트] | 복잡한 패턴 학습, 어텐션 메커니즘 |

## 향후 확장 방향
- 다양한 5G 시나리오(실내/실외, 고밀도/저밀도)에 대한 모델 적용
- 다중 KPI 동시 예측을 위한 멀티태스크 학습 접근법

## 프로젝트 소스코드

**GitHub Repository**: https://github.com/nkim71-dev/kpiEstimate


### 구조

```
kpi-estimation/
├── data/
│   ├── raw/                    # 원본 5G 데이터셋
│   ├── processed/              # 전처리된 데이터
│   └── columns.json            # 데이터 컬럼 설명
├── src/
│   ├── preprocessData.py       # 데이터 전처리 스크립트
│   ├── trainModel.py           # 모델 학습 스크립트
│   └── inferenceModel.py       # 모델 추론 스크립트
├── models/                     # 학습된 모델 가중치 (h5 파일)
├── figures/                    # 예측 결과 시각화
├── requirements.txt            # 필요 패키지 목록
└── README.md                   # 프로젝트 설명
```

### 실행 방법

#### 1. 환경 설정
```bash
# Conda 환경 생성 (Python 3.11)
conda create -n env python=3.11
conda activate env

# 패키지 설치
pip install -r requirements.txt

# 오류 발생 시
conda install pip
pip install --upgrade pip
```

#### 2. 데이터 전처리
```bash
python src/preprocessData.py
```
- 5G 데이터셋 전처리 수행
- **[NOTE]** Data parsing will be shortly added

#### 3. 모델 학습
```bash
# Dense 모델 학습 (기본값)
python src/trainModel.py

# Transformer 모델 학습
python src/trainModel.py --model-name transformer
```
- 학습된 모델은 `models` 폴더에 h5 파일로 저장
- 파일명은 생성 시간을 기준으로 관리

#### 4. 모델 추론 및 평가
```bash
# Dense 모델 추론 (기본값)
python src/inferenceModel.py

# Transformer 모델 추론
python src/inferenceModel.py --model-name transformer
```
- RMSE 및 MAE 계산 및 출력
- 실제 KPI와 예측 KPI 비교 시각화
- 결과는 `figures` 폴더에 저장


## 참고 자료

- **5G Dataset**: https://github.com/uccmisl/5Gdataset
- Attention Is All You Need (Vaswani et al., 2017) - Transformer 원론

---

**작성자**: [이름]  
**작성일**: YYYY-MM-DD  
