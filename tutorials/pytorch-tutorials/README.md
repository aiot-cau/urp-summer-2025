# PyTorch 프레임워크 튜토리얼

이 디렉토리에는 AIoT 연구에 활용할 수 있는 PyTorch 프레임워크 튜토리얼이 포함되어 있습니다. PyTorch는 유연하고 직관적인 딥러닝 프레임워크로, 동적 계산 그래프와 명확한 Python 스타일의 코드를 제공합니다.

## 학습 목표

1. PyTorch 프레임워크의 기본 구조와 철학 이해
2. 텐서 조작 및 자동 미분 시스템 활용 방법 습득
3. 다양한 신경망 모델 설계, 학습, 평가 기법 익히기
4. AIoT 응용을 위한 효율적인 모델 개발 및 최적화 능력 함양

## 튜토리얼 구성

### 1. PyTorch 기초 (basics)

- `01_tensors_and_autograd.ipynb`: 텐서 연산 및 자동 미분 시스템
- `02_neural_network_basics.ipynb`: 신경망 구성 요소와 기본 구조
- `03_datasets_and_dataloaders.ipynb`: 데이터 로딩 및 전처리
- `04_training_validation_loop.ipynb`: 기본적인 학습 및 검증 루프 구현

### 2. 신경망 모델 구현 (models)

- `01_linear_models.ipynb`: 선형 모델 및 다층 퍼셉트론
- `02_convolutional_networks.ipynb`: 합성곱 신경망(CNN) 구현
- `03_recurrent_networks.ipynb`: 순환 신경망(RNN, LSTM, GRU) 구현
- `04_attention_transformer.ipynb`: 어텐션 메커니즘 및 트랜스포머 구현

### 3. 모델 학습 및 최적화 (training)

- `01_optimizers.ipynb`: 다양한 최적화 알고리즘 활용
- `02_loss_functions.ipynb`: 손실 함수의 선택과 구현
- `03_regularization.ipynb`: 과적합 방지 기법 (Dropout, BatchNorm, L1/L2)
- `04_learning_rate_scheduling.ipynb`: 학습률 스케줄링 기법

### 4. 고급 기능 및 도구 (advanced)

- `01_custom_modules.ipynb`: 커스텀 모듈 및 레이어 구현
- `02_model_saving_loading.ipynb`: 모델 저장 및 불러오기
- `03_tensorboard_integration.ipynb`: TensorBoard를 활용한 모니터링
- `04_distributed_training.ipynb`: 분산 학습 및 병렬 처리

### 5. AIoT 응용 (aiot_applications)

- `01_time_series_forecasting.ipynb`: 시계열 센서 데이터 예측
- `02_anomaly_detection.ipynb`: 이상 탐지 모델 구현
- `03_model_quantization.ipynb`: 모델 양자화 및 경량화
- `04_deployment_tools.ipynb`: TorchScript, ONNX 활용한 배포 기법

## 사전 요구사항

- Python 기초 지식
- 딥러닝 및 신경망의 기본 개념 이해
- NumPy 사용 경험
- 선형대수, 미적분학 기초 지식

## 환경 설정

튜토리얼을 진행하기 위한 환경 설정 방법입니다:

```bash
# 가상 환경 생성 및 활성화
conda create -n pytorch-tutorial python=3.8
conda activate pytorch-tutorial

# PyTorch 설치 (CUDA 지원 버전, 11.6 기준)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# CPU 전용 버전 (GPU가 없는 경우)
# pip install torch torchvision torchaudio

# 필수 패키지 설치
pip install numpy pandas matplotlib scikit-learn jupyterlab

# 추가 패키지 설치
pip install seaborn plotly tqdm tensorboard
```

## 튜토리얼 활용 방법

1. 순서대로 각 노트북을 실행하며 개념과 구현 방법 학습
2. 코드를 직접 수정하고 실험해보며 이해도 향상
3. 각 노트북 끝부분의 실습 과제 수행
4. 학습한 내용을 실제 AIoT 프로젝트에 응용

## PyTorch 주요 개념 요약

### 1. 텐서(Tensor)와 자동 미분(Autograd)

PyTorch의 핵심은 텐서와 자동 미분 시스템입니다:

```python
# 텐서 생성
import torch
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x**2 + 2*x + 1

# 역전파 계산
y.backward(torch.ones_like(y))

# 기울기 확인
print(x.grad)  # tensor([4., 6., 8.])
```

### 2. 모듈(Module)과 모델 구현

PyTorch 모델은 `nn.Module`을 상속하여 구현합니다:

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 3. 데이터 로딩 및 전처리

Dataset과 DataLoader를 사용해 데이터를 효율적으로 처리합니다:

```python
from torch.utils.data import Dataset, DataLoader

# 커스텀 데이터셋
class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 데이터로더 생성
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 4. 학습 루프 구현

PyTorch에서는 명시적인 학습 루프를 구현합니다:

```python
# 학습 루프 예시
def train(model, dataloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## 실습 프로젝트 아이디어

학습한 내용을 응용할 수 있는 AIoT 관련 프로젝트 아이디어:

1. **웨어러블 센서 데이터 분석**: 가속도계 데이터를 활용한 활동 인식 모델
2. **스마트 홈 전력 소비 예측**: 가전 기기별 전력 소비 패턴 예측 모델
3. **산업 장비 상태 모니터링**: 다중 센서 데이터 기반 장비 상태 진단 시스템
4. **환경 센서 이상 탐지**: 온도, 습도, 먼지 농도 등 환경 센서 이상값 감지 모델

## PyTorch 모델 저장 및 불러오기

모델을 저장하고 불러오는 주요 방법:

```python
# 모델 저장하기
torch.save(model.state_dict(), 'model_weights.pth')  # 가중치만 저장
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
}, 'checkpoint.pth')  # 체크포인트 저장

# 모델 불러오기
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth'))

# 체크포인트 불러오기
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
```

## 모델 배포 및 최적화

AIoT 환경에서 PyTorch 모델을 효율적으로 배포하는 방법:

1. **TorchScript 변환**: 모델을 C++에서 실행 가능한 형태로 변환
   ```python
   scripted_model = torch.jit.script(model)
   scripted_model.save('scripted_model.pt')
   ```

2. **ONNX 변환**: 다양한 플랫폼에서 실행 가능한 ONNX 형식으로 변환
   ```python
   dummy_input = torch.randn(1, input_size)
   torch.onnx.export(model, dummy_input, 'model.onnx')
   ```

3. **양자화**: 모델 크기 감소 및 추론 속도 향상
   ```python
   quantized_model = torch.quantization.quantize_dynamic(
       model, {nn.Linear}, dtype=torch.qint8
   )
   ```

## 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch by Eli Stevens, Luca Antiga, and Thomas Viehmann](https://www.manning.com/books/deep-learning-with-pytorch)
- [fastai 라이브러리 및 강의](https://www.fast.ai/)

## 도움 받기

튜토리얼 진행 중 어려움이 있을 경우:
1. 연구실 Slack 채널 #pytorch-help에 질문
2. 지도교수 또는 선배 연구자에게 문의
3. [PyTorch 포럼](https://discuss.pytorch.org/) 활용

---

## PyTorch 모델 개발 체크리스트

1. **데이터 준비**
   - Dataset 클래스 구현
   - 데이터 변환 및 정규화
   - DataLoader 구성

2. **모델 설계**
   - nn.Module 서브클래스 구현
   - 레이어 구성 및 forward 메서드 정의
   - 초기화 전략 선택

3. **학습 설정**
   - 손실 함수 선택
   - 옵티마이저 구성
   - 학습률 스케줄러 설정 (필요시)

4. **학습 및 검증**
   - 학습/검증 루프 구현
   - 성능 지표 모니터링
   - 체크포인트 저장

5. **모델 평가 및 개선**
   - 테스트 세트 평가
   - 파라미터 튜닝
   - 모델 아키텍처 개선

6. **모델 최적화 및 배포**
   - 모델 경량화 (양자화, 가지치기)
   - TorchScript/ONNX 변환
   - 엣지 디바이스 배포 테스트