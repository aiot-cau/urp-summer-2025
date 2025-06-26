# Python 개발 환경 설정 가이드

AIoT Lab 연구 활동을 위한 Python 개발 환경 설정 가이드로, 이 문서에서는 Anaconda를 사용한 환경 설정 방법과 주요 라이브러리 설치 방법을 안내합니다.
(Note: Colab 사용 시 별도의 환경 설정은 필요하지 않습니다.)

## 목차

1. [Anaconda 설치](#1-anaconda-설치)
2. [가상 환경 생성](#2-가상-환경-생성)
3. [주요 라이브러리 설치](#3-주요-라이브러리-설치)
4. [Jupyter Notebook 설정](#4-jupyter-notebook-설정)
5. [IDE 설정 (VSCode)](#5-ide-설정-vscode)
6. [Git 연동](#6-git-연동)
7. [환경 공유 및 재현](#7-환경-공유-및-재현)

## 1. Anaconda 설치

Anaconda는 Python과 다양한 과학 계산용 패키지를 포함하는 배포판으로, 가상 환경 관리에 용이

### 설치 단계

1. [Anaconda 다운로드 페이지](https://www.anaconda.com/products/individual#Downloads)에서 운영체제에 맞는 버전 다운로드
2. 다운로드한 설치 파일 실행
3. 설치 과정에서 "Add Anaconda to my PATH environment variable" 옵션 체크 권장
4. 설치 완료 후 Anaconda Prompt(또는 터미널)에서 다음 명령어로 설치 확인:
   ```bash
   conda --version
   python --version
   ```

## 2. 가상 환경 생성

프로젝트별로 독립된 환경을 사용하여 라이브러리 버전 충돌을 방지

### AIoT Lab 기본 환경 설정

```bash
# 'aiot' 이름의 가상 환경 생성 (Python 3.8 기준)
conda create -n aiot python=3.8

# 가상 환경 활성화
conda activate aiot

# 가상 환경 비활성화 (작업 완료 후)
conda deactivate
```

## 3. 주요 라이브러리 설치 (설치 예시로 필요에 따라 버전)

* AIoT Lab 연구에 필요한 주요 라이브러리 설치
* 아래 예시는 대표적인 구성이며, 프로젝트 필요에 따라 라이브러리 종류나 버전을 조정하여 설치해 사용


### 가상환경 활성화 및 기본 데이터 분석과 과학 계산 라이브러리 설치
```bash
conda activate aiot
conda install numpy pandas matplotlib seaborn scikit-learn scipy
```

### 딥러닝 프레임워크 - PyTorch 설치
* 설치 시 CUDA 버전에 맞는 버전 설치 필요
```bash
# CUDA 지원 GPU가 있는 경우 (CUDA 11.3 기준)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# CPU 전용 버전
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 딥러닝 프레임워크 - Keras/TensorFlow 설치
* 설치 시 CUDA 버전에 맞는 버전 설치 필요
```bash
conda install tensorflow
```

### 기타 유용한 라이브러리
* pip: Python 기본 패키지 관리자
```bash
pip install tqdm  # 진행 상황 표시
pip install plotly  # 인터랙티브 시각화
pip install ipywidgets  # Jupyter 위젯
pip install openpyxl  # Excel 파일 처리
```

## 4. Jupyter Notebook 설정

Jupyter Notebook은 코드 실행, 시각화, 문서화를 동시에 할 수 있는 도구

```bash
# Jupyter 설치
conda install jupyter notebook

# Jupyter Lab 설치 (고급 기능 제공)
conda install jupyterlab

# 실행
jupyter notebook  # 또는 jupyter lab
```


## 5. IDE 설정 (VSCode)

Visual Studio Code는 가볍고 강력한 코드 에디터로 Python 개발에 적합

1. [VS Code 다운로드](https://code.visualstudio.com/) 및 설치
2. Python 확장 프로그램 설치 (Extensions 탭에서 'Python' 검색)
3. 기타 유용한 확장 프로그램 설치 (Extensions 탭)

### VS Code에서 Anaconda 환경 사용하기

1. `Ctrl+Shift+P` (또는 `Cmd+Shift+P`) 단축키로 명령 팔레트 열기
2. "Python: Select Interpreter" 검색
3. 생성한 conda 환경 선택 (예: 'aiot')

## 6. Git 연동

연구 결과를 관리하고 공유하기 위해 Git 설정

```bash
# Git 설치 확인
git --version

# Git 기본 설정
git config --global user.name "이름"
git config --global user.email "이메일"

# 저장소 복제
git clone https://github.com/username/repository.git

# Git에 포함하지 않을 항목을 .gitignore에 추가
echo "env/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

## 7. 환경 공유 및 재현

연구 결과의 재현성을 위한 환경 설정 공유

### 환경 내보내기

```bash
# 현재 환경의 패키지 목록을 requirements.txt로 저장
pip freeze > requirements.txt

# Conda 환경 내보내기
conda env export > environment.yml
```

### 환경 재현하기

```bash
# pip로 패키지 설치
pip install -r requirements.txt

# Conda 환경 재현
conda env create -f environment.yml
```

## 문제 해결

* 라이브러리 충돌 시: `conda install 패키지명=버전` 형태로 특정 버전 지정 설치
* CUDA 오류: GPU 드라이버와 CUDA 버전이 호환되는지 확인
* 메모리 부족: 대용량 데이터셋 처리 시 `batch_size` 축소 또는 데이터 샘플링 고려

## 참고 자료

* [Anaconda 공식 문서](https://docs.anaconda.com/)
* [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
* [TensorFlow 공식 문서](https://www.tensorflow.org/learn)
* [VS Code Python 튜토리얼](https://code.visualstudio.com/docs/python/python-tutorial)

---
