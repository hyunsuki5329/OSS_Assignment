# 프로젝트 제출 요약 코멘트 템플릿

아래 템플릿을 복사해서 PR 코멘트 또는 과제 제출 코멘트로 사용하세요.

## 1) 과제 개요

- 과제명: [과제명 입력]
- 저장소: [저장소 URL 입력]
- 작업 브랜치: [브랜치명 입력]
- 사용 언어/라이브러리: Python, scikit-learn, matplotlib

## 2) 구현 내용

- 사용 데이터셋: load_breast_cancer
- 데이터 분할: train/test split (예: 8:2, random_state=42)
- 학습 모델:
  - Logistic Regression
  - Random Forest (100 trees)
  - Random Forest (300 trees)
  - Random Forest (depth=5)
- 평가 항목:
  - Accuracy
  - Precision / Recall / F1
  - ROC-AUC

## 3) 결과 요약

- 최고 성능 모델: [모델명 입력]
- Accuracy: [값 입력]
- ROC-AUC: [값 입력]
- 핵심 해석:
  - [결과 해석 1]
  - [결과 해석 2]

## 4) 첨부 산출물

- 보고서: report.md
- 이미지:
  - report_assets/dataset_image.png
  - report_assets/label_mask_image.png
  - report_assets/prediction_image.png
  - report_assets/accuracy_graph.png
  - report_assets/confusion_matrix.png
  - report_assets/roc_curve.png

## 5) 실행 방법

```powershell
& ".venv/Scripts/python.exe" -m pip install -r requirements.txt
& ".venv/Scripts/python.exe" -m pip install matplotlib
& ".venv/Scripts/python.exe" breast_cancer_classifier.py
& ".venv/Scripts/python.exe" generate_report_assets.py
```

## 6) 커밋 요약

- [커밋 1 메시지] - [핵심 변경]
- [커밋 2 메시지] - [핵심 변경]
- [커밋 3 메시지] - [핵심 변경]

## 7) 확인 요청

- 데이터 로드/분할/학습/예측/정확도 출력 요구사항 충족 여부 확인 부탁드립니다.
- 보고서와 이미지 산출물 포함 여부 확인 부탁드립니다.