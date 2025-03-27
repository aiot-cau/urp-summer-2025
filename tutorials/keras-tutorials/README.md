# Keras 프레임워크 튜토리얼

이 디렉토리에는 AIoT 연구에 활용할 수 있는 Keras 프레임워크 튜토리얼이 포함되어 있습니다. Keras는 TensorFlow 위에서 동작하는 고수준 딥러닝 API로, 사용하기 쉽고 빠르게 모델을 구현할 수 있습니다.

## 학습 목표

1. Keras 프레임워크의 기본 구조와 원리 이해
2. 다양한 딥러닝 모델 구현 방법 습득
3. 모델 학습, 평가, 저장 및 불러오기 방법 익히기
4. AIoT 응용을 위한 커스텀 모델 개발 능력 함양

## 튜토리얼 구성

### 1. Keras 기초 (basics)

- `01_keras_introduction.ipynb`: Keras 소개 및 기본 사용법
- `02_sequential_api.ipynb`: Sequential API를 사용한 모델 구성
- `03_functional_api.ipynb`: Functional API를 사용한 복잡한 모델 구성
- `04_model_subclassing.ipynb`: Model 서브클래싱을 통한 커스텀 모델 구현

### 2. 핵심 레이어 및 모델 (layers_models)

- `01_dense_layers.ipynb`: 완전 연결 레이어와 MLP 모델
- `02_cnn_layers.ipynb`: 합성곱 레이어와 CNN 모델
- `03_rnn_layers.ipynb`: 순환 레이어(SimpleRNN, LSTM, GRU)
- `04_attention_transformer.ipynb`: 어텐션 메커니즘과 트랜스포머 구현

### 3. 모델 학습 및 평가 (training_evaluation)

- `01_compile_fit.ipynb`: 모델 컴파일 및 학습
- `02_callbacks.ipynb`: 콜백 함수를 활용한 학습 제어
- `03_custom_training_loop.ipynb`: 커스텀 학습 루프 구현
- `04_model_evaluation.ipynb`: 모델 평가 방법과 지표

### 4. 고급 기법 (advanced)

- `01_data_generators.ipynb`: 데이터 제너레이터와 데이터 증강
- `02_transfer_learning.ipynb`: 전이 학습 및 파인 튜닝
- `03_custom_layers.ipynb`: 커스텀 레이어 구현
- `04_tensorboard.ipynb`: TensorBoard를 활용한 모델 모니터링

### 5. AIoT 응용 (aiot_applications)

- `01_time_series_prediction.ipynb`: 시계열 센서 데이터 예측
- `02_anomaly_detection.ipynb`: 센서 데이터 이상 탐지
- `03_model_optimization.ipynb`: 모델 경량화 및 최적화
- `04_deployment_strategies.ipynb`: 엣지 디바이스 배포 전략

## 사전 요구사항

- Python 기초 지식
- 딥러닝의 기본 개념 이해
- NumPy, Pandas, Matplotlib 사용 경험
- Linux/macOS 환경 권장 (Windows에서도 가능)

## 환경 설정

튜토리얼을 진행하기 위한 환경 설정 방법입니다:

```bash
# 가상 환경 생성 및 활성화 
conda create -n keras-tutorial python=3.8
conda activate keras-tutorial

# TensorFlow와 Keras 설치
pip install tensorflow

# 필수 패키지 설치
pip install numpy pandas matplotlib scikit-learn jupyterlab

# 추가 패키지 설치
pip install seaborn plotly tqdm pydot graphviz
```

## 튜토리얼 활용 방법

1. 각 주제별 노트북 파일을 순서대로 실행하며 학습
2. 코드를 직접 수정하고 실행해보며 개념 이해
3. 각 노트북 끝부분의 실습 과제 수행
4. 학습한 내용을 실제 AIoT 프로젝트에 적용

## 실습 프로젝트

튜토리얼 학습 후 다음과 같은 프로젝트를 시도해 볼 수 있습니다:

1. **스마트홈 온도 예측 시스템**: 시계열 데이터를 활용한 온도/습도 예측 모델
2. **산업 장비 이상 탐지**: 여러 센서 데이터를 활용한 이상 탐지 모델
3. **웨어러블 활동 인식**: 가속도 센서 데이터를 이용한 사용자 활동 인식
4. **에너지 소비 최적화**: 전력 사용량 예측 및 최적화 모델

## 모델 저장 및 불러오기

Keras에서 모델을 저장하고 불러오는 기본 방법입니다:

```python
# 모델 저장하기
model.save('my_model.h5')  # 전체 모델 저장
model.save_weights('my_model_weights.h5')  # 가중치만 저장

# 모델 불러오기
from tensorflow import keras
loaded_model = keras.models.load_model('my_model.h5')
model.load_weights('my_model_weights.h5')
```

## Keras 주요 컴포넌트

### 1. 모델 유형
- **Sequential**: 층을 순차적으로 쌓는 간단한 모델
- **Functional API**: 복잡한 모델 토폴로지 구성 가능
- **Model 서브클래싱**: 완전한 커스터마이징 가능

### 2. 레이어 종류
- **Dense**: 완전 연결 레이어
- **Conv1D/Conv2D/Conv3D**: 합성곱 레이어
- **MaxPooling/AveragePooling**: 풀링 레이어
- **SimpleRNN/LSTM/GRU**: 순환 레이어
- **Dropout/BatchNormalization**: 정규화 레이어
- **Embedding**: 임베딩 레이어

### 3. 손실 함수
- **MSE/MAE**: 회귀 문제
- **Binary/Categorical CrossEntropy**: 분류 문제
- **Custom Loss**: 사용자 정의 손실 함수

### 4. 옵티마이저
- **SGD**: 확률적 경사 하강법
- **Adam/RMSprop**: 적응형 학습률 옵티마이저
- **Custom Optimizer**: 사용자 정의 옵티마이저

## 참고 자료

- [Keras 공식 문서](https://keras.io/)
- [TensorFlow 공식 튜토리얼](https://www.tensorflow.org/tutorials)
- [Deep Learning with Python by François Chollet](https://www.manning.com/books/deep-learning-with-python)
- [TensorFlow 2.0 Tutorial by Aurélien Géron](https://github.com/ageron/tf2_course)

## 도움 받기

튜토리얼 진행 중 어려움이 있을 경우:
1. 연구실 Slack 채널 #keras-help에 질문
2. 지도교수 또는 선배 연구자에게 문의
3. [Stack Overflow](https://stackoverflow.com/questions/tagged/keras) Keras 태그 검색

---

## Keras 모델 개발 체크리스트

1. **데이터 준비**
   - 데이터 수집 및 정리
   - 전처리 및 정규화
   - 훈련/검증/테스트 세트 분할

2. **모델 설계**
   - 적절한 모델 아키텍처 선택
   - 레이어 구성 및 파라미터 설정
   - 활성화 함수 선택

3. **모델 컴파일**
   - 적합한 손실 함수 선택
   - 옵티마이저 및 학습률 설정
   - 평가 지표 정의

4. **모델 학습**
   - 배치 크기 및 에폭 수 설정
   - 콜백 함수 구성
   - 학습 과정 모니터링

5. **모델 평가 및 개선**
   - 테스트 세트에서 성능 평가
   - 하이퍼파라미터 튜닝
   - 오류 분석 및 모델 개선

6. **모델 배포**
   - 모델 저장 및 변환
   - 엣지 디바이스 최적화
   - 배포 파이프라인 구축