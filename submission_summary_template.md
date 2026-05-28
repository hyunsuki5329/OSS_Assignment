# 프로젝트 제출 요약 코멘트 템플릿

아래 템플릿을 복사해서 PR 코멘트 또는 과제 제출 코멘트로 사용하세요.

## 1) 과제 개요

- 과제명: GitHub 활용 OSS 실습 - load_breast_cancer 분류 모델 구현
- 저장소: https://github.com/hyunsuki5329/OSS_Assignment
- 작업 브랜치: oss-assignment-mainline
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

- 최고 성능 모델: Logistic Regression
- Accuracy: 0.982456
- ROC-AUC: 0.995370
- 핵심 해석:
  - load_breast_cancer 데이터셋에서는 Logistic Regression이 가장 높은 정확도와 안정적인 분류 성능을 보였습니다.
  - Random Forest는 파라미터를 조정해도 본 실험 설정에서는 Logistic Regression보다 낮은 정확도를 보였습니다.

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

- Initial upload: breast cancer classifier scaffold - 데이터 로드/분할/학습/평가 기본 코드 업로드
- Train model: compare logistic regression and random forest - 모델 비교 구조 추가
- Evaluation: print sample predictions alongside accuracy - 정확도와 샘플 예측 출력 보강
- Parameter change experiment: tune random forest variants - Random Forest 파라미터 실험 추가
- Report: add final project report with figures and metrics - 보고서 및 시각화 산출물 추가
- Docs: add submission summary comment template - 제출 요약 템플릿 추가

## 7) 확인 요청

- 데이터 로드/분할/학습/예측/정확도 출력 요구사항 충족 여부 확인 부탁드립니다.
- 보고서와 이미지 산출물 포함 여부 확인 부탁드립니다.