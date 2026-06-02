import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# 1. 데이터 불러오기
try:
    df = pd.read_csv('clinical_data.csv', sep='\t')
    print("✅ clinical_data.csv 파일을 성공적으로 로드했습니다.")
except FileNotFoundError:
    print("❌ 에러: 'clinical_data.csv' 파일이 같은 폴더에 없습니다! 다운로드 후 다시 시도하세요.")
    exit()

# 2. 타겟 변수(SUBTYPE) 결측치 제거 및 분리
df = df.dropna(subset=['SUBTYPE'])
X = df.drop(columns=['SUBTYPE'])
y = df['SUBTYPE']

# 3. 과적합 및 행정용 변수 대거 제거 (R 코드와 100% 일치)
target_leakage_cols = ["TUMOR_TYPE", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE", "ICD_O_3_HISTOLOGY", "OS_STATUS", "OS_MONTHS", "DSS_STATUS", "DSS_MONTHS", "DFS_STATUS", "DFS_MONTHS", "PFS_STATUS", "PFS_MONTHS", "DAYS_LAST_FOLLOWUP", "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT", "PERSON_NEOPLASM_CANCER_STATUS", "PATIENT_ID", "OTHER_PATIENT_ID", "SAMPLE_ID", "FORM_COMPLETION_DATE", "TISSUE_SOURCE_SITE_CODE"]
X = X.drop(columns=[col for col in target_leakage_cols if col in X.columns])

# 4. 결측치 비율이 30%를 초과하는 열 제거
na_ratio = X.isna().mean()
X = X.loc[:, na_ratio <= 0.3]

# 수치형 변수와 범주형 변수 자동 분류
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 5. 전처리 파이프라인 매커니즘 구축 (R의 caret과 동기화)
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

full_processor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('variance_filter', VarianceThreshold(threshold=0.0))
])

# 6. 학습 및 테스트 데이터 분할 (7:3, R 코드 시드 123 고정 반영)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

print("⚙️ 최신 라이브러리 버전으로 모델 학습을 시작합니다…")

# 7. 최적화 모델 학습 및 파이프라인 결합
# (1) Random Forest (ntree=500 반영)
rf_pipeline = Pipeline(steps=[('process', full_processor),
                             ('classifier', RandomForestClassifier(n_estimators=500, random_state=123))])
rf_pipeline.fit(X_train, y_train)

# (2) Naive Bayes
nb_pipeline = Pipeline(steps=[('process', full_processor),
                             ('classifier', GaussianNB())])
nb_pipeline.fit(X_train, y_train)

# (3) Logistic Regression (최신 버전 규격에 맞게 multi_class 옵션 제거 및 C값 반영)
lr_pipeline = Pipeline(steps=[('process', full_processor),
                             ('classifier', LogisticRegression(C=1/0.01, max_iter=1000, random_state=123))])
lr_pipeline.fit(X_train, y_train)

# 8. 최종 .pkl 파일로 저장하여 웹앱에 넘겨주기
with open('random_forest_model.pkl', 'wb') as f: pickle.dump(rf_pipeline, f)
with open('naive_bayes_model.pkl', 'wb') as f: pickle.dump(nb_pipeline, f)
with open('logistic_regression_model.pkl', 'wb') as f: pickle.dump(lr_pipeline, f)

# 웹앱에서 입력 폼을 동적으로 구성하기 위한 피처 리스트 저장
with open('clinical_features.pkl', 'wb') as f:
    pickle.dump({'num_cols': num_cols, 'cat_cols': cat_cols, 'classes': list(rf_pipeline.classes_)}, f)

print("🎉 [성공] 세 가지 임상 ML 모델 파일(.pkl)이 안전하게 추출되었습니다!")
