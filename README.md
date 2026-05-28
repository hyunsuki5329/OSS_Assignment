# OSS_Assignment

scikit-learn의 load_breast_cancer 데이터셋을 사용해 유방암 분류 모델을 학습하고 평가하는 과제 프로젝트입니다.

## 과제 충족 항목

- 데이터셋 로드: load_breast_cancer
- 데이터 분할: train_test_split (학습/평가 분리)
- 모델 학습: 분류 모델 학습 수행
- 예측 및 정확도: 테스트 데이터 예측 후 정확도 출력

## 구현 내용

- Logistic Regression 모델 학습 및 정확도 측정
- Random Forest 모델(여러 파라미터) 학습 및 정확도 비교
- 가장 성능이 좋은 모델 선택
- 테스트 데이터 일부(10개)에 대한 예측값/실제값 출력

## 실행 환경

- Python 3.12
- scikit-learn

의존성 설치:

```powershell
& ".venv/Scripts/python.exe" -m pip install -r requirements.txt
```

## 실행 방법

```powershell
& ".venv/Scripts/python.exe" breast_cancer_classifier.py
```

## 출력 예시 항목

- Logistic Regression accuracy
- Random Forest (100 trees) accuracy
- Random Forest (300 trees) accuracy
- Random Forest (depth=5) accuracy
- Best model
- Sample predictions (상위 10개)

## 파일 구성

- breast_cancer_classifier.py: 데이터 로드, 분할, 학습, 평가 전체 코드
- requirements.txt: 실행에 필요한 패키지 목록

## Git 작업 요약

- 과제 브랜치: oss-assignment
- 의미 있는 커밋 단위로 구현/학습/평가/파라미터 실험/문서화를 분리하여 기록
