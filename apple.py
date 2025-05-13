import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# 1. 설정
image_folder = '../project'  # apple / not_apple 하위 폴더 존재해야 함
allowed_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')

image_data = []
labels = []

# 이미지 회전 함수
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

# 2. 이미지 불러오기 + 전처리
for class_folder in os.listdir(image_folder):
    class_path = os.path.join(image_folder, class_folder)
    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):
        if filename.endswith(allowed_exts):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img_rotated = rotate_image(img, 345)
            img_flipped = cv2.flip(img_rotated, 1)
            img_resized = cv2.resize(img_flipped, (128, 128))
            img_flat = img_resized.flatten()

            image_data.append(img_flat)
            labels.append(class_folder)  # 'apple' 또는 'not_apple'

# 3. 데이터프레임 & 라벨 인코딩
df = pd.DataFrame(image_data)
df['label'] = labels

le = LabelEncoder()
y = le.fit_transform(df['label'])  # apple → 0, not_apple → 1 (예시)
X = df.drop(columns=['label'])

# 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. 모델 정의 & 파라미터 탐색
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# 6. 평가
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f" 최종 모델 정확도: {acc:.4f}")
print("분류 리포트:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 7. 새로운 이미지로 예측
test_image_path = '../project/apple/Apple_10.jpg'  

img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
if img is not None:
    img_rotated = rotate_image(img, 345)
    img_flipped = cv2.flip(img_rotated, 1)
    img_resized = cv2.resize(img_flipped, (128, 128))
    img_flat = img_resized.flatten().reshape(1, -1)

    prediction = best_model.predict(img_flat)
    result = le.inverse_transform(prediction)[0]

    print(f"이 이미지는 '{result}' 입니다.")
else:
    print(" 예측 이미지 불러오기 실패:", test_image_path)
