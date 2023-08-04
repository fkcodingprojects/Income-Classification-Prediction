#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)


# ### Naive Bayes Prediction Model 

# In[1]:


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

# Naive Bayes (Gaussian Naive Bayes)
print('---- Naive Bayes (Gaussian Naive Bayes) ----')
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(f'Accuracy: {accuracy_nb:.4f}')
print(f'Classification Report:\n{report_nb}')
print(f'Confusion Matrix:\n{cm_nb}')


# ### Missing Values -> Naive Bayes Prediction Model 

# In[2]:


# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Data Preprocessing
# Handle Missing Values with Mean Imputation
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

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes (Gaussian Naive Bayes)
print('---- Naive Bayes (Gaussian Naive Bayes) ----')
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(f'Accuracy: {accuracy_nb:.4f}')
print(f'Classification Report:\n{report_nb}')
print(f'Confusion Matrix:\n{cm_nb}')


# ####  Missing Values -> Naive Bayes Prediction Model with Standard Scaler

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Replace 'path/to/income_classification.csv' with the actual path to your file

# Data Preprocessing
# Handle Missing Values with Mean Imputation
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

# Feature Scaling for Numerical Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Naive Bayes (Gaussian Naive Bayes)
print('---- Naive Bayes (Gaussian Naive Bayes) ----')
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(f'Accuracy: {accuracy_nb:.4f}')
print(f'Classification Report:\n{report_nb}')
print(f'Confusion Matrix:\n{cm_nb}')


# ####  Missing Values -> Naive Bayes Prediction Model with Grid Search

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Data Preprocessing
# Handle Missing Values with Mean Imputation
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

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter Tuning for Naive Bayes
param_grid = {}  # No hyperparameters to tune for Naive Bayes

# Naive Bayes (Gaussian Naive Bayes) with GridSearchCV
print('---- Naive Bayes (Gaussian Naive Bayes) with GridSearchCV ----')
naive_bayes = GaussianNB()
grid_search = GridSearchCV(naive_bayes, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print('Best model:', grid_search.best_estimator_)

# Evaluate the model on the test set
best_nb = grid_search.best_estimator_
y_pred_nb = best_nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(f'Accuracy: {accuracy_nb:.4f}')
print(f'Classification Report:\n{report_nb}')
print(f'Confusion Matrix:\n{cm_nb}')

