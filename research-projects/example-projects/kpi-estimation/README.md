# KPI Estimation 프로젝트

## 프로젝트 소개

KPI Estimation 프로젝트는 시계열 데이터 분석과 머신러닝/딥러닝 기법을 활용하여 비즈니스 핵심 성과 지표(KPI, Key Performance Indicators)를 예측하는 연구입니다. 이 프로젝트는 AIoT Lab의 시계열 예측 기술을 활용한 실제 응용 사례를 보여줍니다.

## 목표

- 다양한 시계열 예측 모델의 성능을 비교 분석
- 비즈니스 KPI 데이터에 최적화된 예측 모델 개발
- 시계열 데이터의 특성(계절성, 추세, 이벤트 등)을 고려한 모델링 방법 탐구
- 예측 불확실성을 추정하고 의사결정에 활용 가능한 형태로 제공

## 데이터셋

이 프로젝트에서는 [GitHub 저장소](https://github.com/nkim71-dev/kpiEstimate)에서 제공하는 샘플 KPI 데이터셋을 활용합니다:

- 일별 매출 데이터
- 고객 확보 비용
- 신규 사용자 수
- 활성 사용자 수
- 해지율
- 마케팅 지출
- 특별 이벤트 및 캠페인 정보

## 방법론

이 프로젝트에서는 다음과 같은 방법론을 활용합니다:

1. **데이터 전처리 및 특성 공학**
   - 시계열 분해(Decomposition): 추세, 계절성, 잔차 분석
   - 결측치 및 이상치 처리
   - 시간 기반 특성 생성(요일, 월, 휴일 등)
   - 윈도우 기반 롤링 통계량 생성

2. **모델링 접근법**
   - 통계적 모델: ARIMA, ETS, Prophet
   - 머신러닝 모델: Random Forest, XGBoost, SVR
   - 딥러닝 모델: LSTM, GRU, Transformer
   - 앙상블 모델: 여러 모델의 예측 결합

3. **모델 학습 및 평가**
   - 시간 기반 교차 검증
   - 다양한 예측 구간(1일, 7일, 30일) 평가
   - 평가 지표: MAE, RMSE, MAPE, CRPS

## 프로젝트 구조

```
kpiEstimation/
├── data/
│   ├── raw/              # 원본 KPI 데이터
│   └── processed/        # 전처리된 데이터
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb    # 데이터 탐색
│   ├── 02_data_preprocessing.ipynb      # 데이터 전처리
│   ├── 03_statistical_models.ipynb      # 통계적 모델 실험
│   ├── 04_ml_models.ipynb               # 머신러닝 모델 실험
│   └── 05_dl_models.ipynb               # 딥러닝 모델 실험
├── src/
│   ├── data_preprocessing.py            # 데이터 전처리 함수
│   ├── feature_engineering.py           # 특성 공학 함수
│   ├── models/                          # 모델 구현
│   │   ├── statistical_models.py        # 통계적 모델
│   │   ├── ml_models.py                 # 머신러닝 모델
│   │   └── dl_models.py                 # 딥러닝 모델
│   ├── evaluation.py                    # 모델 평가 함수
│   └── visualization.py                 # 시각화 함수
├── results/
│   ├── model_comparison.csv             # 모델 성능 비교 결과
│   └── figures/                         # 생성된 그래프
└── README.md                            # 프로젝트 설명
```

## 실행 방법

```bash
# 1. 저장소 복제 (원본 KPI Estimation 저장소 참조)
git clone https://github.com/nkim71-dev/kpiEstimate.git
cd kpiEstimate

# 2. 가상 환경 설정
conda create -n kpi-env python=3.8
conda activate kpi-env
pip install -r requirements.txt

# 3. 노트북 실행
jupyter notebook notebooks/
```

## 주요 실험 결과

아래는 다양한 모델의 30일 KPI 예측 성능 비교입니다:

| 모델 | MAE | RMSE | MAPE |
|------|-----|------|------|
| ARIMA | 0.245 | 0.312 | 8.7% |
| Prophet | 0.201 | 0.278 | 7.2% |
| Random Forest | 0.189 | 0.256 | 6.5% |
| XGBoost | 0.172 | 0.238 | 5.8% |
| LSTM | 0.156 | 0.215 | 5.3% |
| Transformer | 0.143 | 0.198 | 4.8% |
| Ensemble | 0.128 | 0.182 | 4.2% |

## 학습 내용

이 프로젝트를 통해 다음과 같은 AIoT 관련 지식과 기술을 배울 수 있습니다:

1. **시계열 데이터 분석 기법**
   - 시계열 데이터의 특성 이해
   - 시계열 시각화 및 패턴 분석
   - 시계열 전처리 및 특성 공학

2. **다양한 예측 모델의 이해**
   - 통계적 모델의 원리와 한계
   - 머신러닝/딥러닝 기반 시계열 모델링
   - 모델 앙상블 및 하이브리드 접근법

3. **모델 평가 및 튜닝**
   - 시계열 특화 평가 방법론
   - 하이퍼파라미터 최적화
   - 모델 불확실성 정량화 방법

4. **실무 응용 능력**
   - 비즈니스 지표에 대한 이해
   - 예측 모델의 실제 응용 방법
   - 의사결정 지원을 위한 데이터 분석

## 향후 확장 방향

이 프로젝트는 다음과 같은 방향으로 확장할 수 있습니다:

- 여러 KPI를 동시에 예측하는 다변량 모델 개발
- 외부 데이터(경제 지표, 소셜 미디어 데이터 등) 통합
- 실시간 업데이트 및 온라인 학습 시스템 구축
- 설명 가능한 AI 기법을 통한 예측 결과 해석 개선

## 참고 자료

- [원본 GitHub 저장소](https://github.com/nkim71-dev/kpiEstimate)
- DeepAR: Probabilistic forecasting with autoregressive recurrent networks (Salinas et al., 2020)
- N-BEATS: Neural basis expansion analysis for interpretable time series forecasting (Oreshkin et al., 2020)
- Forecasting at scale with Facebook Prophet (Taylor & Letham, 2018)

## 연락처

프로젝트 관련 문의: nkim71@cau.ac.kr

---

*이 프로젝트는 AIoT Lab 학부 인턴 프로그램의 예시 프로젝트로 제공됩니다.*