#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)


# ### Logistic Regression Prediction Model  

# In[9]:


# Data Preprocessing
# Convert categorical columns to numerical using One-Hot Encoding
categorical_cols = [' workclass', ' education', ' marital-status', ' occupation', ' relationship', ' race', ' sex', ' native-country']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Encode the target variable 'income'
label_encoder = LabelEncoder()
df_encoded[' income'] = label_encoder.fit_transform(df_encoded[' income'])

# Split the data into features (X) and the target variable (y)
X = df_encoded.drop(columns=[' income'])
y = df_encoded[' income']

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
print('---- Logistic Regression ----')
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f'Accuracy: {accuracy_lr:.4f}')
print(f'Classification Report:\n{report_lr}')
print(f'Confusion Matrix:\n{cm_lr}')


# ### Missing Values ->  Logistic Regression Prediction Model  

# In[10]:


# Data Preprocessing
# Handling Missing Values
df.fillna(df.mean(), inplace=True)

# Convert categorical columns to numerical using One-Hot Encoding
categorical_cols = [' workclass', ' education', ' marital-status', ' occupation', ' relationship', ' race', ' sex', ' native-country']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Encode the target variable 'income'
label_encoder = LabelEncoder()
df_encoded[' income'] = label_encoder.fit_transform(df_encoded[' income'])

# Split the data into features (X) and the target variable (y)
X = df_encoded.drop(columns=[' income'])
y = df_encoded[' income']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Logistic Regression
print('---- Logistic Regression ----')
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f'Accuracy: {accuracy_lr:.4f}')
print(f'Classification Report:\n{report_lr}')
print(f'Confusion Matrix:\n{cm_lr}')


# ### Missing Values ->  Logistic Regression Prediction Model  with L2 Regularisation (Series)

# In[16]:


# Data Preprocessing
# Handling Missing Values
df.fillna(df.mean(), inplace=True)

# Convert categorical columns to numerical using One-Hot Encoding
categorical_cols = [' workclass', ' education', ' marital-status', ' occupation', ' relationship', ' race', ' sex', ' native-country']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Encode the target variable 'income'
label_encoder = LabelEncoder()
df_encoded[' income'] = label_encoder.fit_transform(df_encoded[' income'])

# Split the data into features (X) and the target variable (y)
X = df_encoded.drop(columns=[' income'])
y = df_encoded[' income']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Loop through a series of C values
C_values = [0.001, 0.01, 0.1, 1, 10]

for C in C_values:
    print(f'---- Logistic Regression with L2 Regularization (C={C}) ----')
    logistic_regression = LogisticRegression(penalty='l2', C=C)
    logistic_regression.fit(X_train, y_train)
    y_pred_lr = logistic_regression.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    print(f'Accuracy: {accuracy_lr:.4f}')
    print(f'Classification Report:\n{report_lr}')
    print(f'Confusion Matrix:\n{cm_lr}\n')


# ### Missing Values with SMOTE ->  Logistic Regression Prediction Model  

# In[5]:


from imblearn.over_sampling import SMOTE

# Data Preprocessing
# Handling Missing Values
df.fillna(df.mean(), inplace=True)

# Convert categorical columns to numerical using One-Hot Encoding
categorical_cols = [' workclass', ' education', ' marital-status', ' occupation', ' relationship', ' race', ' sex', ' native-country']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Encode the target variable 'income'
label_encoder = LabelEncoder()
df_encoded[' income'] = label_encoder.fit_transform(df_encoded[' income'])

# Split the data into features (X) and the target variable (y)
X = df_encoded.drop(columns=[' income'])
y = df_encoded[' income']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handling Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the resampled data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Logistic Regression
print('---- Logistic Regression ----')
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f'Accuracy: {accuracy_lr:.4f}')
print(f'Classification Report:\n{report_lr}')
print(f'Confusion Matrix:\n{cm_lr}')

