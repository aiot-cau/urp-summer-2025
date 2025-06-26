# Fork 기반 GitHub 협업 가이드
* Git은 분산 버전 관리 시스템으로, 코드 변경 사항을 추적하고 여러 개발자의 작업을 조율하는 데 사용
* **urp-summer-2025** 저장소의 경우, 학생 계정은 직접적으로 push 권한이 없기 때문에 다음 흐름에 따라 작업

## Git 설치 및 기본 설정

### GitHub 계정 설정

1. [GitHub](https://github.com/) 웹사이트에서 계정 생성
2. 프로필 설정 및 SSH 키 등록 (선택 사항)
3. 연구실 GitHub 조직에 가입 요청

### 설치

- Windows: [Git for Windows](https://gitforwindows.org/) 다운로드 및 설치
- macOS: 터미널에서 `git --version` 실행 (설치되어 있지 않은 경우 설치 안내 따름)
- Linux: `sudo apt-get install git` 또는 `sudo yum install git`

### 기본 설정
   ```bash
   # 사용자 정보 설정
   git config --global user.name "이름"
   git config --global user.email "이메일"
   ```

## 협업 흐름 요약

| 단계 | 설명 |
|------|------|
| 독립적 저장소로 fork | 원본 저장소를 자신의 개인 계정으로 `fork` (직접 수정 권한이 없는 경우 사용하는 방식) |
| 로컬 복제 및 upstream 등록 | 자신의 GitHub 저장소를 로컬 컴퓨터에 `clone`하고, 원본 저장소를 `upstream`으로 연결 |
| 원본 저장소와 동기화 | 작업 전에 원본 저장소의 최신 상태를 로컬 저장소에 반영 (충돌 방지) |
| 브랜치 생성 및 작업 | main branch를 그대로 두고, 작업 전용 `branch`를 만들어 개발|
| 변경 사항 업로드 | 작업한 내용을 자신의 GitHub 저장소에 `push` (개인 저장소에 업로드)|
| Pull Request (PR) 생성 | GitHub에서 원본 저장소에 작업 `pull request` (관리자가 승인 후 `merge`) |

### 1. 저장소 Fork 및 로컬 복제 (최초 1회)

1. GitHub에서 원본 저장소(`aiot-cau/urp-summer-2025`)로 이동
2. 오른쪽 상단의 `Fork` 버튼 클릭
3. 자신의 계정 또는 조직에 Fork 생성
4. 로컬에 복제:

   ```bash
   git clone https://github.com/your-username/urp-summer-2025.git
   cd urp-summer-2025
   ```

5. 원본 저장소를 `upstream` 으로 등록:

   ```bash
   git remote add upstream https://github.com/aiot-cau/urp-summer-2025.git
   ```


### 2. 작업 전 원본 저장소 동기화 (upstream 반영)

Fork는 원본 저장소와 자동으로 동기화되지 않으므로, **작업을 시작하기 전에 반드시 최신 상태를 반영**

   ```bash
   # upstream에서 최신 변경 사항 가져오기
   git fetch upstream

   # 내 main 브랜치로 이동 후 병합
   git checkout main
   git merge upstream/main
   ```

> ⚠️ 충돌이 발생하는 경우 수동으로 해결 후 커밋 필요

### 3. 브랜치 생성, 작업 및 업로드 (개인 저장소)

충돌이 발생하거나 작업 내용이 덮어씌어졌을 시, **branch 생성 없이 main에서 바로 작업한 경우 복구가 어려울 수 있음**

1. 기능/수정 작업을 위한 브랜치 생성:

   ```bash
   git checkout -b branch-name
   ```

2. 코드 수정 및 테스트 진행
3. 변경 사항 저장:

   ```bash
   git add .
   git commit -m "구현 내용 요약"
   ```

4. 자신의 저장소에 push:

   ```bash
   git push origin branch-name
   ```


### 4. Pull Request 생성

1. GitHub 웹사이트에서 `Compare & pull request` 버튼 클릭
2. PR 제목 및 설명 작성
3. 대상 브랜치: `aiot-cau/urp-summer-2025`의 `main`
4. PR 생성 후 승인 기다리기



## 기타

### 기본 Git 명령어

* 저장소 초기화 및 복제
   ```bash
   # 새 저장소 초기화
   git init

   # 원격 저장소 복제
   git clone https://github.com/username/repository.git
   ```

* 변경 사항 관리
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

* 브랜치 관리
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

* 원격 저장소 작업

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

* 커밋 내용 취소 (**실수로 커밋한 경우**)
  ```bash
  # 마지막 커밋 수정
  git commit --amend
  
  # 마지막 커밋 취소 (변경 사항 유지)
  git reset HEAD~1
  ```

### 추가 학습 자료

- [Git 공식 문서](https://git-scm.com/doc)
- [GitHub 학습 리소스](https://docs.github.com/en/get-started)
- [Git 브랜칭 모델](https://nvie.com/posts/a-successful-git-branching-model/)
- [대화형 Git 학습](https://learngitbranching.js.org/)

### 연구실 Git/GitHub 사용 규칙
1. 개인 API 키, 비밀번호 등 민감한 정보는 절대 커밋하지 않기
2. 대용량 데이터 파일은 `.gitignore`에 추가하고 별도 저장소 활용
3. 주기적으로 진행 상황 업데이트 및 커밋하기
4. 실험 결과와 해당 코드 버전 연결되도록 관리하기

---
