# 24619025

# 1.드라이브 마운트 및 데이터 압축 해제
from google.colab import drive
drive.mount('/content/drive')
!unzip --qq /content/drive/MyDrive/open.zip -d dataset

# 2. 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
!pip install imbalanced-learn

# 3. 데이터 불러오기 및 기본 분할
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
submission = pd.read_csv('dataset/sample_submission.csv')
X = train.drop(columns=['ID', 'Cancer'])
y = train['Cancer']
x_test = test.drop(columns=['ID'])

# 4. 범주형 변수 라벨 인코딩
categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    x_test[col] = le.transform(x_test[col])

# 5. 수치형 변수 스케일링 (표준화)
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

# 6. 학습/검증 데이터 분할 및 SMOTE 오버샘플링
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 7. XGBoost 모델 정의 및 검증 F1 Score 평가
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.2,
    min_child_weight=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_sm, y_train_sm)
val_pred = model.predict(X_val)
val_f1 = f1_score(y_val, val_pred)
print(f"Validation F1 Score (with SMOTE): {val_f1:.4f}")

# 8. 전체 학습 데이터로 재학습 및 예측 파일 생성
X_full, y_full = smote.fit_resample(X, y)
model.fit(X_full, y_full)
final_pred = model.predict(x_test)
submission['Cancer'] = final_pred
submission.to_csv('xgb_smote_optimized_submit.csv', index=False)

# 9. 제출 파일 다운로드
from google.colab import files
files.download('xgb_smote_optimized_submit.csv')

# 최종 제출
![image](https://github.com/user-attachments/assets/8907f23c-8e22-4db5-9a99-55bfb22fdc9e)
