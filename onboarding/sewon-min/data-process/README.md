# Pytorch 데이터 처리
- Dataset: 데이터셋 전체를 표현하며 __len__, __getitem__ 구현 필요 
- 매번 정의해줘야하나? => 커스텀 데이터셋이면 YES
- DataLoader: Dataset을 순회 가능한(minibatch 단위) 객체로 감싸줌


# 데이터 로딩
```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```
- train = True, False 정도의 차이만 있음

# 사용자 정의 `Dataset` 클래스 만들기
```python
from torch.utils.data import Dataset
import os, pandas as pd
from torchvision.io import read_image

# 사용자 정의 데이터셋 클래스
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        초기화 함수
        - annotations_file: 이미지 파일명과 라벨이 저장된 CSV 파일 경로
        - img_dir: 이미지 파일들이 저장된 디렉토리 경로
        - transform: 이미지에 적용할 transform 함수 (예: ToTensor(), Normalize 등)
        - target_transform: 라벨에 적용할 transform 함수
        """
        # CSV 파일을 읽어 DataFrame으로 저장 (컬럼명 지정)
        self.img_labels = pd.read_csv(annotations_file, names=["file_name", "label"])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        전체 샘플 수를 반환
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 샘플(이미지, 라벨)을 반환
        - idx: 샘플 인덱스
        """
        # 이미지 파일 경로 구성
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # 이미지를 텐서로 읽어오기
        image = read_image(img_path)
        
        # 라벨 가져오기
        label = self.img_labels.iloc[idx, 1]
        
        # 이미지에 transform 적용 (선택적)
        if self.transform:
            image = self.transform(image)
        
        # 라벨에 transform 적용 (선택적)
        if self.target_transform:
            label = self.target_transform(label)
        
        # 이미지와 라벨을 딕셔너리로 반환
        return {"image": image, "label": label}

```
- `read_image` => 이미지를 tensor로 변환해서 가져옴
- `img_labels` => csv 데이터에서 정답 값만 읽어옴(인덱스 1이 label이라저렇게 쓴 것) - idx == 행, 1 == 열

# 근데 위에서 정답값(label)에 왜 transform??
정답(label)에 `transform`을 거는 이유는 **정답값이 모델이 바로 사용하기 어려운 형태일 수 있기 때문**이에요. 아래에 주요 이유를 정리해줄게요.

---

### ✅ 왜 정답(label)에 `target_transform`을 적용하나?

#### 1. **클래스 이름 → 숫자 인덱스로 변환할 때**

예를 들어, 정답이 `"cat"`, `"dog"`처럼 문자열인 경우, 모델이 이해할 수 있도록 숫자로 바꿔야 합니다.

```python
target_transform = lambda x: {"cat": 0, "dog": 1}[x]
```

#### 2. **정답을 One-hot 벡터로 변환할 때**

모델 출력이 softmax이고 one-hot 레이블이 필요한 경우:

```python
target_transform = lambda x: F.one_hot(torch.tensor(x), num_classes=10)
```

#### 3. **정답값의 dtype 변환이 필요할 때**

예를 들어 `float`로 바꿔야 하는 회귀 문제라면:

```python
target_transform = lambda x: torch.tensor(x, dtype=torch.float)
```

---

### 📌 정리

* `transform`: 입력 데이터(X)에 적용 → 예: `ToTensor()`, `Normalize()`
* `target_transform`: 정답(y)에 적용 → 예: 인덱싱, one-hot, 형변환 등

> 즉, 모델이 학습하기 좋은 **정답 형태로 가공**하는 게 `target_transform`의 목적입니다.

---

예를 들어, 정답이 `"ankle_boot"`처럼 문자열일 경우 숫자로 바꾸는 transform을 쓸 수 있겠죠:

```python
label_map = {"tshirt": 0, "trouser": 1, ..., "ankle_boot": 9}
target_transform = lambda y: label_map[y]
```

필요 없는 경우는 `None`으로 두면 됩니다.


# 그럼 이건 이미지에 쓰이는 transform이랑 다른거네 normalization 이런건 아니니까

### ✅ 핵심 정리

| 항목                 | 목적                              | 예시 transform                                       | 적용 대상   |
| ------------------ | ------------------------------- | -------------------------------------------------- | ------- |
| `transform`        | **이미지 전처리** (모델 입력에 적합하게 만듦)    | `ToTensor()`, `Normalize(mean, std)`, `Resize()` 등 | `image` |
| `target_transform` | **정답(라벨) 전처리** (모델 학습에 적합한 형태로) | `lambda x: one_hot(x)`, `int(x)`, `label_map[x]` 등 | `label` |

---

### 📌 예시 비교

#### 이미지용 transform (입력 X)

```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

#### 정답용 target\_transform (정답 y)

```python
target_transform = lambda y: torch.tensor(y, dtype=torch.long)
```

---

### ❗ 둘은 목적과 적용 대상이 완전히 다릅니다

* `transform`은 **이미지 데이터를 텐서로 바꾸고 정규화(Normalization)** 해서 **모델 입력**으로 사용하기 좋게 만드는 과정이고,
* `target_transform`은 **정답 레이블이 텍스트이거나, 포맷이 맞지 않을 때 가공하는 보조 수단**이에요.

---

필요 없으면 `None`으로 설정해도 되고, 둘 다 쓰는 경우도 많음

```python
CustomImageDataset(..., transform=transform, target_transform=target_transform)
```

이렇게 쓰면 이미지와 라벨 둘 다 적절히 전처리되어 모델에 입력될 수 있죠.

# DataLoader

 
### ✅ 역할

`DataLoader`는 `Dataset`으로부터 데이터를 **효율적으로 꺼내주는 도구**입니다.

### ✅ 주요 기능

* 데이터를 **미니배치 단위(batch)** 로 불러옴
* **에폭마다 섞기(shuffle)** 지원
* **병렬 처리(multiprocessing)** 로 로딩 속도 향상
* 반복 가능한 객체(iterable)로 사용 가능 → `for batch in loader:` 형태

---

### 🧪 기본 사용법

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
```

* `batch_size=64`: 한 번에 64개 샘플 반환
* `shuffle=True`: 매 epoch마다 데이터 순서 섞음

---

### 🔁 순회 예시 (학습 루프용)

```python
for batch in train_dataloader:
    images, labels = batch
    # 모델에 입력: model(images)
```

또는 한 번만 꺼내볼 땐:

```python
images, labels = next(iter(train_dataloader))
print(images.shape)  # torch.Size([64, 1, 28, 28])
print(labels.shape)  # torch.Size([64])
```

---

### 👀 시각화 예시

```python
img = images[0].squeeze()
label = labels[0]
plt.imshow(img, cmap="gray")
plt.title(f"Label: {label}")
plt.show()
```

---

### 📝 요약

| 기능            | 설명                       |
| ------------- | ------------------------ |
| `batch_size`  | 한 번에 반환할 데이터 수           |
| `shuffle`     | 에폭마다 순서 섞기               |
| `num_workers` | 데이터 로딩에 사용할 subprocess 수 |
| `drop_last`   | 마지막 남은 불완전 배치 버릴지 여부     |

---