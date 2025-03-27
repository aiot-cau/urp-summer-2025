# Git 및 GitHub 기초 가이드

## Git이란?

Git은 분산 버전 관리 시스템으로, 코드 변경 사항을 추적하고 여러 개발자의 작업을 조율하는 데 사용됩니다. 이 가이드에서는 연구 활동에 필요한 Git의 기본 사용법을 안내합니다.

## GitHub 계정 설정

1. [GitHub](https://github.com/) 웹사이트에서 계정 생성
2. 프로필 설정 및 SSH 키 등록 (선택 사항)
3. 연구실 GitHub 조직에 가입 요청

## Git 설치 및 기본 설정

### 설치

- Windows: [Git for Windows](https://gitforwindows.org/) 다운로드 및 설치
- macOS: 터미널에서 `git --version` 실행 (설치되어 있지 않은 경우 설치 안내 따름)
- Linux: `sudo apt-get install git` 또는 `sudo yum install git`

### 기본 설정

```bash
# 사용자 정보 설정
git config --global user.name "홍길동"
git config --global user.email "이메일@example.com"

# 편집기 설정 (선택 사항)
git config --global core.editor "code --wait"  # VS Code 사용 시
```

## 기본 Git 명령어

### 저장소 초기화 및 복제

```bash
# 새 저장소 초기화
git init

# 원격 저장소 복제
git clone https://github.com/username/repository.git
```

### 변경 사항 관리

```bash
# 변경 사항 확인
git status

# 변경 사항 스테이징
git add 파일명
git add .  # 모든 변경 사항 스테이징

# 변경 사항 커밋
git commit -m "커밋 메시지"

# 변경 내역 확인
git log
```

### 브랜치 관리

```bash
# 브랜치 목록 확인
git branch

# 새 브랜치 생성
git branch 브랜치명

# 브랜치 전환
git checkout 브랜치명
# 또는
git switch 브랜치명  # Git 2.23 이상

# 브랜치 생성 및 전환
git checkout -b 브랜치명
```

### 원격 저장소 작업

```bash
# 원격 저장소 추가
git remote add origin https://github.com/username/repository.git

# 변경 사항 업로드
git push origin 브랜치명

# 원격 저장소 변경 사항 가져오기
git pull origin 브랜치명

# 원격 저장소 정보 확인
git remote -v
```

## GitHub 작업 흐름

### 1. 저장소 포크 및 복제

1. GitHub에서 연구실 저장소를 자신의 계정으로 포크(Fork)
2. 포크한 저장소를 로컬에 복제:
   ```bash
   git clone https://github.com/your-username/repository.git
   ```

### 2. 브랜치 생성 및 작업

1. 새 기능/작업을 위한 브랜치 생성:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. 코드 수정 및 실험 진행
3. 변경 사항 커밋:
   ```bash
   git add .
   git commit -m "구현 내용 설명"
   ```

### 3. 변경 사항 업로드 및 Pull Request

1. 변경 사항을 자신의 GitHub 저장소에 업로드:
   ```bash
   git push origin feature/your-feature-name
   ```
2. GitHub 웹사이트에서 Pull Request 생성

## 협업을 위한 Git 사용 팁

1. **커밋 메시지 작성 규칙**:
   - 첫 줄은 50자 이내로 변경 사항 요약
   - 필요시 빈 줄 후 상세 설명 추가
   - 현재형 동사로 시작 (예: "Add", "Fix", "Update")

2. **자주 커밋하기**:
   - 논리적으로 구분된 작은 단위로 커밋
   - 하나의 커밋은 하나의 기능/수정에 집중

3. **Pull Request 작성 요령**:
   - 제목: 변경 사항 명확히 표현
   - 내용: 무엇을, 왜 변경했는지 설명
   - 관련 이슈 연결 (있는 경우)

4. **코드 리뷰 에티켓**:
   - 건설적인 피드백 제공
   - 코드가 아닌 사람을 비판하지 않기
   - 의견 불일치 시 근거와 대안 제시

## Git 관련 유용한 도구

- **GitHub Desktop**: GUI 기반 Git 클라이언트
- **GitKraken**: 시각적 Git 관리 도구
- **VS Code Git 확장**: 에디터 내에서 Git 작업 수행

## 문제 해결

- **충돌(Conflict) 해결**: 같은 파일의 같은 부분을 여러 사람이 수정할 때 발생
  ```bash
  # 충돌 해결 후
  git add .
  git commit -m "Resolve merge conflicts"
  ```

- **실수로 커밋한 경우**:
  ```bash
  # 마지막 커밋 수정
  git commit --amend
  
  # 마지막 커밋 취소 (변경 사항 유지)
  git reset HEAD~1
  ```

## 연구실 Git/GitHub 사용 규칙

1. 주요 연구 코드는 항상 버전 관리하기
2. 대용량 데이터 파일은 `.gitignore`에 추가하고 별도 저장소 활용
3. 개인 API 키, 비밀번호 등 민감한 정보는 절대 커밋하지 않기
4. 매주 진행 상황 업데이트 및 커밋하기
5. 실험 결과와 해당 코드 버전 연결되도록 관리하기

## 추가 학습 자료

- [Git 공식 문서](https://git-scm.com/doc)
- [GitHub 학습 리소스](https://docs.github.com/en/get-started)
- [Git 브랜칭 모델](https://nvie.com/posts/a-successful-git-branching-model/)
- [대화형 Git 학습](https://learngitbranching.js.org/)