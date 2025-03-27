# 딥러닝 기초 튜토리얼

이 디렉토리에는 AIoT 연구에 필요한 딥러닝 기초 이론과 개념을 학습할 수 있는 튜토리얼이 포함되어 있습니다.

## 학습 목표

1. 딥러닝의 기본 개념 및 원리 이해
2. 주요 신경망 아키텍처 학습
3. 딥러닝 모델 학습 및 평가 방법 습득
4. AIoT 응용을 위한 딥러닝 기술 적용

## 튜토리얼 구성

### 1. 딥러닝 기초 (fundamentals)

- `01_neural_networks_intro.ipynb`: 신경망 기본 개념 및 역사
- `02_activation_functions.ipynb`: 활성화 함수의 종류와 특징
- `03_loss_functions.ipynb`: 손실 함수의 이해와 선택
- `04_backpropagation.ipynb`: 역전파 알고리즘 이해

### 2. 주요 신경망 아키텍처 (architectures)

- `01_feedforward_networks.ipynb`: 기본 피드포워드 신경망
- `02_convolutional_networks.ipynb`: 합성곱 신경망(CNN)
- `03_recurrent_networks.ipynb`: 순환 신경망(RNN, LSTM, GRU)
- `04_transformer_networks.ipynb`: 트랜스포머 아키텍처

### 3. 모델 학습 및 최적화 (training)

- `01_gradient_descent.ipynb`: 경사 하강법 알고리즘
- `02_optimization_techniques.ipynb`: 최적화 기법(Adam, RMSprop 등)
- `03_regularization.ipynb`: 과적합 방지 기법(Dropout, L1/L2 등)
- `04_hyperparameter_tuning.ipynb`: 하이퍼파라미터 튜닝 방법

### 4. 모델 평가 및 분석 (evaluation)

- `01_evaluation_metrics.ipynb`: 성능 평가 지표
- `02_model_interpretation.ipynb`: 모델 해석 기법
- `03_visualization_techniques.ipynb`: 모델 시각화 방법
- `04_ablation_studies.ipynb`: 모델 구성 요소 분석

### 5. AIoT 응용을 위한 딥러닝 (applications)

- `01_time_series_forecasting.ipynb`: 시계열 예측 모델
- `02_anomaly_detection.ipynb`: 이상 탐지 기법
- `03_sensor_data_classification.ipynb`: 센서 데이터 분류
- `04_lightweight_models.ipynb`: 엣지 디바이스용 경량 모델

## 사전 요구사항

- Python 기초 지식
- 선형대수학, 미적분학, 확률통계 기본 개념
- NumPy, Pandas, Matplotlib 사용 경험
- PyTorch 또는 TensorFlow/Keras 기초 이해

## 튜토리얼 환경 설정

튜토리얼을 진행하기 위한, **딥러닝 환경 설정 방법**입니다:

```bash
# 가상 환경 생성 및 활성화
conda create -n dl-tutorial python=3.8
conda activate dl-tutorial

# 필수 패키지 설치
pip install numpy pandas matplotlib jupyter

# PyTorch 설치 (GPU 지원 버전)
pip install torch torchvision torchaudio

# TensorFlow 설치 (선택사항)
pip install tensorflow

# 추가 패키지 설치
pip install scikit-learn tqdm plotly seaborn
```

## 튜토리얼 활용 방법

1. 각 주제별 노트북(`*.ipynb`)을 순서대로 학습
2. 노트북의 코드 셀을 실행하며 내용 이해
3. 각 노트북 마지막 부분에 있는 과제 수행
4. 실습 프로젝트를 통한 학습 내용 응용

## 실습 프로젝트

튜토리얼 학습 후 다음과 같은 실습 프로젝트를 수행해 볼 수 있습니다:

1. **온도/습도 예측 모델**: 시계열 센서 데이터를 활용한 환경 변수 예측
2. **장비 이상 탐지 시스템**: 다양한 센서 데이터를 활용한 이상 탐지 모델 개발
3. **활동 분류기**: 가속도계 데이터를 이용한 사용자 활동 분류
4. **에너지 소비 최적화**: 시계열 예측을 통한 에너지 사용량 최적화

## 참고 자료

### 교재 및 책
- Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron
- Deep Learning with Python by François Chollet

### 온라인 자료
- [Stanford CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [DeepLearning.AI 강좌](https://www.deeplearning.ai/)
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [TensorFlow 공식 튜토리얼](https://www.tensorflow.org/tutorials)

### 논문 모음
- [딥러닝 필독 논문 목록](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)

## 도움받기

튜토리얼 진행 중 어려움이 있을 경우:
1. 연구실 Slack 채널 #dl-help에 질문 게시
2. 지도교수 또는 선배 연구자에게 문의
3. 매주 수요일 딥러닝 스터디 모임 참여

---

## 딥러닝 개념 요약

### 인공 신경망 기본 구조
- **인공 뉴런(퍼셉트론)**: 입력값에 가중치를 적용, 활성화 함수를 통과시켜 출력 생성
- **다층 퍼셉트론(MLP)**: 여러 층의 뉴런으로 구성된 네트워크
- **은닉층(Hidden Layer)**: 입력층과 출력층 사이에 위치, 복잡한 패턴 학습

### 주요 활성화 함수
- **Sigmoid**: 0~1 사이 값 출력, 그래디언트 소실 문제 있음
- **Tanh**: -1~1 사이 값 출력, Sigmoid보다 기울기 소실 덜함
- **ReLU**: max(0, x), 계산 효율적, 그래디언트 소실 감소
- **Leaky ReLU**: max(0.01x, x), 음수 입력에서도 학습 가능

### 최적화 알고리즘
- **경사 하강법(Gradient Descent)**: 손실 함수의 기울기를 따라 파라미터 조정
- **확률적 경사 하강법(SGD)**: 일부 데이터만 사용하여 파라미터 업데이트
- **Adam**: 적응형 학습률을 사용하는 효율적인 최적화 알고리즘

### 딥러닝 모델 종류
- **CNN**: 이미지 처리에 적합, 합성곱과 풀링 연산 사용
- **RNN/LSTM/GRU**: 순차 데이터 처리, 시계열 예측에 적합
- **Transformer**: 자연어 처리에서 혁신적 성능, 어텐션 메커니즘 활용
- **AutoEncoder**: 비지도 학습, 특징 추출에 유용

### 모델 평가
- **과적합(Overfitting)**: 훈련 데이터에 너무 맞춰져 일반화 성능 저하
- **교차 검증(Cross-validation)**: 다양한 데이터 분할로 일반화 성능 평가
- **정확도, 정밀도, 재현율, F1 점수**: 분류 모델 평가 지표
- **MSE, MAE, RMSE**: 회귀 모델 평가 지표