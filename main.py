import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import os
import sys
from contextlib import contextmanager
import warnings
from sklearn.exceptions import ConvergenceWarning

# Add these warning filters before your model training
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

ews = pd.read_csv("FinancialMarketData_EWS.csv")
X = ews.drop(columns='Y')
y = ews['Y']

# Conduct Isolation Forest training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train['Data'] = pd.to_datetime(X_train['Data'])
X_train['Data'] = X_train['Data'].map(pd.Timestamp.timestamp)
X_test['Data'] = pd.to_datetime(X_test['Data'])
X_test['Data'] = X_test['Data'].map(pd.Timestamp.timestamp)

iso_forest = IsolationForest(n_estimators=100, contamination=0.21, random_state=42)

# Balanced training
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

iso_forest.fit(X_train)
iso_forest.fit(X_train_balanced)

# Predict on the training and testing sets
y_pred_train = iso_forest.predict(X_train)
y_pred_test = iso_forest.predict(X_test)

y_pred_train_balanced = iso_forest.predict(X_train_balanced)
y_pred_test_balanced = iso_forest.predict(X_test)

# Convert predictions: 1 -> 0 (inlier), -1 -> 1 (outlier)
y_pred_train = np.where(y_pred_train == 1, 0, 1)
y_pred_test = np.where(y_pred_test == 1, 0, 1)

y_pred_train_balanced = np.where(y_pred_train_balanced == 1, 0, 1)
y_pred_test_balanced = np.where(y_pred_test_balanced == 1, 0, 1)

print("Train Classification Report:\n", classification_report(y_train, y_pred_train))
print("Test Classification Report:\n", classification_report(y_test, y_pred_test))

print("Train Classification Balanced Report:\n", classification_report(y_train_balanced, y_pred_train_balanced))
print("Test Classification Balanced Report:\n", classification_report(y_test, y_pred_test_balanced))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

# Train logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train['Data'] = pd.to_datetime(X_train['Data'])
X_train['Data'] = X_train['Data'].map(pd.Timestamp.timestamp)
X_test['Data'] = pd.to_datetime(X_test['Data'])
X_test['Data'] = X_test['Data'].map(pd.Timestamp.timestamp)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42, max_iter=1110)
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)

# Standard results
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Fine-tuning
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'], 
    'C': [0.01, 0.1, 1, 10],
    'solver': ['saga'],
    'class_weight': ['balanced', None],
    'l1_ratio': [0, 0.5, 1],
}

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=0
)

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

# Suppress warnings during GridSearchCV
with suppress_output():
    grid_search.fit(X_train_scaled, y_train)

# Get best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test_scaled)

# See results of best trained model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))