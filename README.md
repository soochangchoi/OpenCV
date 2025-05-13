# 🍎 Apple Image Classification Project

과일 이미지(사과 vs 기타 과일)를 분류하기 위한 머신러닝 & 딥러닝 프로젝트입니다.  
Random Forest, XGBoost 모델을 사용하여 이미지 분류 모델을 학습하고,  
Flask API를 통해 웹에서 예측 서비스를 제공합니다.

---

## 📂 프로젝트 구성

| 구분         | 주요 기능                                      | 스크립트                 |
|------------|---------------------------------------------|------------------------|
| ✅ Random Forest  | 흑백 이미지 기반 사과 vs 비사과 분류 모델 학습           | `apple.py`             |
| ✅ XGBoost        | 컬러 이미지 + 증강 기반 XGBoost 분류 모델 학습 및 저장    | `apple_xg_boost.py`    |
| ✅ Flask API     | 웹 API 형태로 이미지 예측 제공 (SVM 모델 기반)            | `apple_predict.py`     |

---

## 🛠 사용 기술

- Python 3.x
- OpenCV, numpy, pandas
- scikit-learn
- XGBoost
- Flask (REST API)
- joblib

---

## ▶ 실행 방법

### 1. Random Forest (흑백 이미지)
```bash
python apple.py
../project/apple vs ../project/not_apple 폴더 이미지 사용

회전, 반전, 리사이즈 (128×128) 전처리

Random Forest + GridSearchCV 최적화

모델 정확도 출력 및 개별 이미지 예측

2. XGBoost (컬러 + 증강 이미지)

복사
편집
python apple_xg_boost.py
다양한 과일 (orange, strawberry, Tomatoes, Pomegranate)과 사과 이미지 사용

증강 (원본, 좌우 반전, 90/180/270도 회전)

64×64 리사이즈 후 XGBoost 모델 학습

xgb_model.pkl 저장

3. Flask API (SVM 모델 사용)

복사
편집
python apple_predict.py
POST 요청으로 이미지 업로드

사전에 저장된 SVM 모델(svm_model.pkl) 기반 분류

예시 요청:

복사
편집
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
