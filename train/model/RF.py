import os
import numpy as np
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from preprocessing import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TIMESTEMP = 70
MAX_EPOCHS = 100

dataPath = '../../data/sample'

pe_df, npe_df = DataLoader().loadWindowPeData(dataPath, TIMESTEMP)

csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

print('< PE data size > \n {}'.format(len(pe_df)))
print('< NPE data size > \n {}'.format(len(npe_df)))


# Divide feature and label
csi_data, csi_label = csi_df.iloc[:, :-1], csi_df.iloc[:, -1]

# Display correlation
# corr = csi_data.corr()
# corr_heatmap(corr)


# Divide Train, Test dataset
X_train, X_test, y_train, y_test = train_test_split(csi_data, csi_label, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, '../../pretrained/std_scaler.pkl')

print('Train: X shape: {}'.format(X_train.shape))
print('Train: y shape: {}'.format(y_train.shape))
print('Valid: X shape: {}'.format(X_valid.shape))
print('Valid: y shape: {}'.format(y_valid.shape))
print('Test: X shape: {}'.format(X_test.shape))
print('Test: y shape: {}'.format(y_test.shape))

inp = (-1, X_train.shape[1], 1)
print('Input shape: {}'.format(inp))

print('X reshape: {}'.format(X_train.shape))

kernel_rf_clf = Pipeline([
    ("svm_clf", RandomForestClassifier(n_estimators=800, max_depth=50, min_samples_leaf=8,
                                       min_samples_split=10, random_state=0, n_jobs=-1))
])

start_time = time.time()
kernel_rf_clf.fit(X_train, y_train)
end_time = time.time()

print(f"학습시간: {end_time - start_time:.5f} sec")

pred = kernel_rf_clf.predict(X_test)
score = kernel_rf_clf.score(X_test, y_test)
print(score)

print(classification_report(y_test, pred))