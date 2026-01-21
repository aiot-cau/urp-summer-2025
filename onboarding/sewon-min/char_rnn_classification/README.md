## 문자 단위 RNN으로 이름 국적 분류하기

### 목표
- 문자열(이름)을 **문자(character) 단위의 시퀀스**로 보고, **순환 신경망(RNN)**을 이용해 해당 이름이 어떤 언어(국적)에서 유래했는지 분류

---

1.  **데이터 전처리: One-Hot 인코딩**
    - 이름에 포함된 각 문자를 고유한 벡터로 변환한다. 예를 들어, `a`는 `[1, 0, 0, ...]`로, `b`는 `[0, 1, 0, ...]`로 표현
    - 이름 전체는 이 **One-Hot 벡터들의 시퀀스(sequence)-나열**로 구성된 3차원 텐서 `[이름 글자 수, 배치 크기, 원핫인코딩된 알파벳 벡터]`로 변환되어 모델의 입력으로 사용

2.  **모델 구조: 기본 RNN**
    - `nn.Linear` 계층을 조합하여 RNN을 직접 구현
    - 모델은 각 문자를 순서대로 입력받아 **은닉 상태(hidden state)**를 계속 업데이트 -> RNN의 특징
    - 마지막 문자가 입력된 후의 최종 은닉 상태를 통해, 18개의 언어 중 어느 것에 속할지에 대한 **확률(LogSoftmax)**을 출력

3.  **학습 과정**
    - **입력**: 이름의 문자 텐서 시퀀스 -> 전처리 필요
    - **정답**: 실제 언어(카테고리)
    - **손실 함수**: `NLLLoss`를 사용하여 모델의 예측과 실제 정답 사이의 오차를 계산
    - **최적화**: 계산된 오차를 바탕으로 **역전파(backpropagation)**를 수행하여 모델의 파라미터를 업데이트하고 점차 정답을 더 잘 맞히도록 학습

## RNN 구조 분석

이 튜토리얼에서 구현된 RNN은 **문자 단위 RNN**으로, 이름의 언어를 분류하는 작업을 수행

### RNN 클래스 구조

```python
<code_block_to_apply_changes_from>
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        # 실제로는 3개의 선형 레이어
        self.i2h = nn.Linear(input_size, hidden_size)    # 입력 → 은닉
        self.h2h = nn.Linear(hidden_size, hidden_size)   # 은닉 → 은닉 (순환!)
        self.h2o = nn.Linear(hidden_size, output_size)   # 은닉 → 출력
    
    def forward(self, input, hidden):
        # 이 부분이 "하나의 레이어"처럼 작동
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
```

### RNN의 주요 특징

1. **입력**: 문자 단위 (57개 문자: a-z, A-Z, 공백, 구두점 등)
2. **은닉층 크기**: 128개 뉴런
3. **출력**: 18개 언어 카테고리 (English, French, German 등)

### 작동 방식

1. **순차 처리**: 이름의 각 문자를 하나씩 입력받음
2. **은닉 상태**: 이전 시간 단계의 정보를 기억
3. **순환 구조**: `h_t = tanh(W_ih * x_t + W_h2h * h_{t-1})`
4. **최종 분류**: 마지막 문자까지 처리한 후 언어 예측

### 네트워크 구성 요소

- **i2h**: 입력 문자를 은닉층으로 변환
- **h2h**: 이전 은닉 상태를 현재 은닉 상태로 변환 (순환 연결)
- **h2o**: 은닉 상태를 출력 카테고리로 변환
- **LogSoftmax**: 출력을 확률 분포로 정규화

이 RNN은 기본적인 "vanilla RNN" 구조로, LSTM이나 GRU보다 단순하지만 문자 단위 언어 분류 작업에 효과적으로 사용

## 이름으로 국적 예측할 때 순환 사용 시점

### 1. **학습 과정에서의 순환**

```python
<code_block_to_apply_changes_from>
```

**예시**: "Albert"라는 이름을 처리할 때
- `i=0`: 'A' + 초기 hidden → 새로운 hidden
- `i=1`: 'l' + 이전 hidden → 새로운 hidden  
- `i=2`: 'b' + 이전 hidden → 새로운 hidden
- `i=3`: 'e' + 이전 hidden → 새로운 hidden
- `i=4`: 'r' + 이전 hidden → 새로운 hidden
- `i=5`: 't' + 이전 hidden → 최종 hidden

### 2. **예측 과정에서의 순환**

```python
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)  # 순환 발생!
    
    return output
```

**예시**: "Dovesky" 예측할 때
- 'D' → hidden1
- 'o' + hidden1 → hidden2  
- 'v' + hidden2 → hidden3
- 'e' + hidden3 → hidden4
- 's' + hidden4 → hidden5
- 'k' + hidden5 → hidden6
- 'y' + hidden6 → 최종 hidden → 언어 예측

### 3. **순환이 발생하는 구체적 시점**

**매 문자마다 순환이 발생합니다:**

```python
# RNN의 forward 함수에서
def forward(self, input, hidden):
    # 이 부분에서 순환이 발생!
    hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
    #                    ↑              ↑
    #                현재 입력        이전 은닉상태 (순환!)
    return output, hidden
```

### 4. **순환의 핵심 역할**

1. **정보 누적**: 각 문자를 처리할 때마다 이전 정보가 은닉 상태에 누적
2. **패턴 인식**: "Albert"에서 'A'부터 't'까지 모든 문자의 패턴을 기억
3. **국적 특성 학습**: 각 언어별 이름의 특징적인 문자 조합을 학습

### 5. **실제 예시**

"Jackson"을 예측할 때:
- 'J' → 영어/스코틀랜드 이름의 첫 글자 패턴 학습
- 'a' + 이전 정보 → 'Ja' 패턴 학습  
- 'c' + 이전 정보 → 'Jac' 패턴 학습
- ... 계속해서 모든 문자를 순차적으로 처리하면서 이전 정보를 활용

**결론**: 순환은 **매 문자를 처리할 때마다** 발생하며, 이전 문자들의 정보를 현재 예측에 활용하는 핵심 메커니즘입니다!

## Hidden이 중요한 이유

### 1. **마지막 출력이 모든 이전 정보를 포함해야 하기 때문**

```python
<code_block_to_apply_changes_from>
```

### 2. **Hidden이 "메모리" 역할을 하기 때문**

**"Albert" 예시로 설명:**

```python
# 각 단계에서 hidden이 이전 정보를 누적
hidden = [0, 0, 0, ...]  # 초기 상태

# 1단계: 'A' 처리
output, hidden = rnn('A', hidden)  # hidden에 'A' 정보 저장
# hidden = [A의 정보, 0, 0, ...]

# 2단계: 'l' 처리  
output, hidden = rnn('l', hidden)  # hidden에 'A' + 'l' 정보 저장
# hidden = [A+l의 정보, 0, 0, ...]

# 3단계: 'b' 처리
output, hidden = rnn('b', hidden)  # hidden에 'A' + 'l' + 'b' 정보 저장
# hidden = [A+l+b의 정보, 0, 0, ...]

# ... 계속

# 마지막: 't' 처리
output, hidden = rnn('t', hidden)  # hidden에 'Albert' 전체 정보 저장
# 이제 output이 "Albert" 전체를 기반으로 한 예측!
```

### 3. **왜 중간 출력을 버리는가?**

```python
for i in range(line_tensor.size()[0]):
    output, hidden = rnn(line_tensor[i], hidden)
    # output은 버리고, hidden만 다음 단계로 전달
    # 왜? output은 현재 문자만 본 결과이지만, 
    # hidden은 모든 이전 문자들의 정보를 포함
```

### 4. **Hidden의 핵심 역할**

**Hidden은 "누적된 컨텍스트"를 담고 있습니다:**

- `hidden[0]`: 첫 번째 문자만 본 상태
- `hidden[1]`: 첫 번째 + 두 번째 문자를 본 상태  
- `hidden[2]`: 첫 번째 + 두 번째 + 세 번째 문자를 본 상태
- ...
- `hidden[마지막]`: **모든 문자를 본 상태** ← 이것이 최종 예측에 사용!

### 5. **실제 예시**

"Jackson"을 예측할 때:
- 중간 단계들: 'J', 'Ja', 'Jac', 'Jack', 'Jacks'만 본 상태
- 마지막 단계: 'Jackson' 전체를 본 상태 → **이때만 의미있는 예측 가능**

### 결론

**Hidden이 중요한 이유:**
1. **정보 누적**: 각 단계에서 이전 정보를 보존
2. **컨텍스트 유지**: 전체 이름의 패턴을 기억
3. **최종 예측**: 마지막 hidden이 모든 정보를 포함하여 정확한 예측 가능

**중간 출력을 버리는 이유:**
- 중간 출력은 불완전한 정보 (일부 문자만 본 상태)
- 마지막 출력만이 전체 이름을 기반으로 한 완전한 예측

따라서 hidden은 **"완전한 정보를 마지막까지 전달하는 메모리"** 역할

## 일반적인 Neural Network vs RNN 비교

### 1. **일반적인 Neural Network 방식**

만약 RNN이 아니라 일반적인 Feed-forward Neural Network였다면:

```python
<code_block_to_apply_changes_from>
```

### 2. **입력 데이터 처리 방식의 차이**

**일반적인 Neural Network:**
```python
# 모든 문자를 한 번에 입력 (고정 크기 필요)
def process_name_fixed(name):
    # 이름을 고정 길이로 패딩 (예: 20자)
    padded_name = name.ljust(20, ' ')  # "Albert" → "Albert              "
    
    # 모든 문자를 one-hot으로 변환하고 평탄화
    # 20자 × 57개 문자 = 1140차원 입력
    input_vector = []
    for char in padded_name:
        char_tensor = letterToTensor(char)  # 57차원
        input_vector.extend(char_tensor)
    
    return torch.tensor(input_vector)  # 1140차원
```

**RNN:**
```python
# 문자를 하나씩 순차적으로 처리
def process_name_rnn(name):
    # 각 문자를 개별적으로 처리
    return [letterToTensor(char) for char in name]  # 57차원씩 순차 처리
```

### 3. **왜 일반적인 Neural Network는 비효율적인가?**

**문제점들:**

1. **고정 크기 제약:**
   ```python
   # "Albert" (6자) vs "Schwarzenegger" (13자)
   # 일반 NN: 둘 다 20자로 패딩해야 함
   # RNN: 길이에 상관없이 처리 가능
   ```

2. **매우 큰 입력 차원:**
   ```python
   # 일반 NN: 20자 × 57차원 = 1140차원 입력
   # RNN: 57차원씩 순차 처리
   ```

3. **엄청나게 많은 파라미터:**
   ```python
   # 일반 NN: 1140 → 128 → 128 → ... → 18
   # 파라미터 수: 1140×128 + 128×128 + ... = 수십만 개
   # RNN: 57×128 + 128×128 + 128×18 = 약 2만 개
   ```

4. **순서 정보 손실:**
   ```python
   # 일반 NN: "Albert"와 "Aelbrt"를 같은 것으로 인식
   # RNN: 순서를 고려하여 처리
   ```

### 4. **실제 비교 예시**

**일반적인 Neural Network 방식:**
```python
# 엄청나게 복잡한 구조 필요
class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        # 1140차원 입력을 처리하기 위한 많은 레이어
        self.fc1 = nn.Linear(1140, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 18)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)
```

**RNN 방식:**
```python
# 간단하고 효율적인 구조
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size)    # 57×128
        self.h2h = nn.Linear(hidden_size, hidden_size)   # 128×128
        self.h2o = nn.Linear(hidden_size, output_size)   # 128×18
```

### 5. **결론**

**일반적인 Neural Network였다면:**
- ✅ **장점**: 병렬 처리 가능, 구현이 직관적
- ❌ **단점**: 
  - 엄청나게 많은 레이어 필요 (수십 개)
  - 고정 크기 제약
  - 순서 정보 손실
  - 과도한 파라미터 수
  - 과적합 위험

**RNN의 장점:**
- ✅ **장점**: 
  - 적은 파라미터로 순서 정보 보존
  - 가변 길이 입력 처리
  - 효율적인 메모리 사용
  - 자연스러운 순차 처리

## RNN의 레이어 구조

### 1. **RNN은 "하나의 레이어"가 반복 사용되는 구조**

```python
<code_block_to_apply_changes_from>
```

### 2. **"하나의 레이어"가 계속 업데이트되는 과정**

**"Albert" 예시:**

```python
# 동일한 RNN 레이어가 6번 반복 사용됨
for i in range(line_tensor.size()[0]):  # 6번 반복
    output, hidden = rnn(line_tensor[i], hidden)
    # ↑ 이 rnn이 "하나의 레이어"처럼 작동
```

**구체적 과정:**
```python
# 1단계: 'A' 처리
hidden = rnn('A', 초기_hidden)  # 같은 rnn 레이어 사용

# 2단계: 'l' 처리  
hidden = rnn('l', hidden)       # 같은 rnn 레이어 사용

# 3단계: 'b' 처리
hidden = rnn('b', hidden)       # 같은 rnn 레이어 사용

# 4단계: 'e' 처리
hidden = rnn('e', hidden)       # 같은 rnn 레이어 사용

# 5단계: 'r' 처리
hidden = rnn('r', hidden)       # 같은 rnn 레이어 사용

# 6단계: 't' 처리
hidden = rnn('t', hidden)       # 같은 rnn 레이어 사용
```

### 3. **일반적인 Neural Network vs RNN 비교**

**일반적인 Neural Network:**
```python
# 여러 개의 다른 레이어들
layer1 = nn.Linear(57, 128)     # 레이어 1
layer2 = nn.Linear(128, 128)    # 레이어 2 (다른 레이어)
layer3 = nn.Linear(128, 128)    # 레이어 3 (다른 레이어)
layer4 = nn.Linear(128, 18)     # 레이어 4 (다른 레이어)

# 각각 다른 파라미터를 가짐
```

**RNN:**
```python
# 하나의 레이어가 반복 사용
rnn_layer = RNN(57, 128, 18)   # 하나의 레이어

# 같은 파라미터로 여러 번 사용
for i in range(6):  # 6번 반복
    output, hidden = rnn_layer(input[i], hidden)
    # ↑ 같은 rnn_layer를 6번 사용
```

### 4. **"하나의 레이어"의 의미**

**RNN의 핵심 아이디어:**
- **파라미터 공유**: 모든 시간 단계에서 같은 가중치 사용
- **상태 업데이트**: hidden state만 계속 업데이트
- **정보 누적**: 이전 정보가 hidden state에 누적

```python
# 시간 단계별로 보면:
# t=0: h_0 = tanh(W_ih * x_0 + W_hh * h_{-1})
# t=1: h_1 = tanh(W_ih * x_1 + W_hh * h_0)    # 같은 W_ih, W_hh 사용!
# t=2: h_2 = tanh(W_ih * x_2 + W_hh * h_1)    # 같은 W_ih, W_hh 사용!
# ...
```

### 5. **시각적 비교**

**일반적인 Neural Network:**
```
입력 → [레이어1] → [레이어2] → [레이어3] → [레이어4] → 출력
        (다른 파라미터) (다른 파라미터) (다른 파라미터) (다른 파라미터)
```

**RNN:**
```
입력1 → [RNN 레이어] → hidden1
입력2 → [RNN 레이어] → hidden2    # 같은 파라미터!
입력3 → [RNN 레이어] → hidden3    # 같은 파라미터!
...
입력N → [RNN 레이어] → hiddenN    # 같은 파라미터!
```

### 결론

**맞습니다! RNN은:**
1. **하나의 레이어**가 반복 사용됨
2. **같은 파라미터**로 여러 번 계산
3. **hidden state**만 계속 업데이트
4. **정보 누적**을 통해 시퀀스 처리







