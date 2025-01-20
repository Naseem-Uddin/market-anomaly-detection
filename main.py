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

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.linear_model import LogisticRegression
import pickle

# Add these warning filters before your model training
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

ews = pd.read_csv("FinancialMarketData_EWS.csv")

features = ['VIX', 'GTITL2YR']

X = ews[features]
y = ews['Y']

# Conduct Isolation Forest training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Tested and ensured best model for f1 score
log_reg = LogisticRegression(random_state=42, max_iter=1100, C=0.01, class_weight="balanced", l1_ratio=0, penalty="l2", solver='liblinear')
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)


# Save the trained model
with open("best_model.pkl", "wb") as file:
    pickle.dump(log_reg, file)

# Save the scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Load the pre-trained best model
with open("best_model.pkl", "rb") as file:
    best_model = pickle.load(file)

# Load the scaler used during training
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# App Title
st.title("Investment Strategy Bot")

# User Input for Ticker
ticker = st.text_input("Enter the stock ticker of the company you want to evaluate:", placeholder="e.g., AAPL")

if ticker:
    st.write(f"Fetching data for {ticker}...")
    try:
        # Fetch historical data from Yahoo Finance
        data = yf.download(ticker, start="2023-01-01", end="2024-12-31")

        if not data.empty:
            st.write("Data fetched successfully!")
            st.dataframe(data.head())

            # Preprocess the data
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            data['Date'] = data['Date'].map(pd.Timestamp.timestamp)

            # Use the same features as during training
            features = ['VIX', 'GTITL2YR']  # Replace with actual training features
            data_scaled = scaler.transform(data[features])  # Scale the matching features

            # Predict using the model
            predictions = best_model.predict(data_scaled)

            # Map predictions to actionable outcomes
            data['Anomaly'] = predictions
            data['Recommendation'] = data['Anomaly'].map({0: "Normal", 1: "Anomaly"})
            data['Strategy'] = data['Recommendation'].map({
                "Anomaly": "Consider Buying",
                "Normal": "Hold",
            })


            st.write("Predictions and strategy:")
            st.dataframe(data[['Date', 'Close', 'Anomaly', 'Recommendation', 'Strategy']])

            # Visualize predictions and strategy
            plt.figure(figsize=(14, 7))
            sns.lineplot(x=data['Date'], y=data['Close'], label='Stock Price')
            sns.scatterplot(
                x=data['Date'], y=data['Close'], 
                hue=data['Strategy'], style=data['Recommendation'],
                palette={'Consider Buying': 'red', 'Hold': 'blue'},
                s=100, legend='full'
            )
            plt.title(f"Investment Strategy for {ticker}")
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            st.pyplot(plt)

        else:
            st.error(f"No data found for ticker {ticker}. Please check the ticker and try again.")

    except Exception as e:
        st.error(f"An error occurred while fetching or processing data: {e}")
else:
    st.write("Please enter a ticker to begin.")