#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)


# ### Ada Boost Prediction Model 

# In[4]:


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

# AdaBoost Classifier
print('---- AdaBoost Classifier ----')
ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)  # You can adjust the number of estimators
ada_boost.fit(X_train, y_train)
y_pred_ab = ada_boost.predict(X_test)
accuracy_ab = accuracy_score(y_test, y_pred_ab)
report_ab = classification_report(y_test, y_pred_ab)
cm_ab = confusion_matrix(y_test, y_pred_ab)
print(f'Accuracy: {accuracy_ab:.4f}')
print(f'Classification Report:\n{report_ab}')
print(f'Confusion Matrix:\n{cm_ab}')


# ### Handling Class Imbalance -> Ada Boost Prediction Model 

# In[3]:


# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

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

# Handling Class Imbalance using RandomOverSampler
sampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

# AdaBoost Classifier
print('---- AdaBoost Classifier ----')
ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_boost.fit(X_resampled, y_resampled)

# Evaluate the model on the test set
y_pred_ab = ada_boost.predict(X_test)
accuracy_ab = accuracy_score(y_test, y_pred_ab)
report_ab = classification_report(y_test, y_pred_ab)
cm_ab = confusion_matrix(y_test, y_pred_ab)

print('---- AdaBoost Classifier with Class Imbalance Handling ----')
print(f'Accuracy: {accuracy_ab:.4f}')
print(f'Classification Report:\n{report_ab}')
print(f'Confusion Matrix:\n{cm_ab}')


# ### Select K Best -> Naive Bayes Prediction Model 

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

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

# Perform Feature Selection with SelectKBest and f_classif
k_best_features = 15  # You can adjust the number of features to select based on your preference
selector = SelectKBest(score_func=f_classif, k=k_best_features)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# AdaBoost Classifier
print('---- AdaBoost Classifier with Feature Selection ----')
ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_boost.fit(X_train_selected, y_train)

# Evaluate the model on the test set
y_pred_ab = ada_boost.predict(X_test_selected)
accuracy_ab = accuracy_score(y_test, y_pred_ab)
report_ab = classification_report(y_test, y_pred_ab)
cm_ab = confusion_matrix(y_test, y_pred_ab)

print(f'Accuracy: {accuracy_ab:.4f}')
print(f'Classification Report:\n{report_ab}')
print(f'Confusion Matrix:\n{cm_ab}')


# ### Recursive Feature Elimination -> Ada Boost Prediction Model 

# In[6]:


# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

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

# Perform Recursive Feature Elimination (RFE) with AdaBoostClassifier
num_features_to_select = 15  # You can adjust the number of features to select based on your preference
ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator=ada_boost, n_features_to_select=num_features_to_select)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Fit the model on the training data with selected features
ada_boost.fit(X_train_rfe, y_train)

# Evaluate the model on the test set
y_pred_ab = ada_boost.predict(X_test_rfe)
accuracy_ab = accuracy_score(y_test, y_pred_ab)
report_ab = classification_report(y_test, y_pred_ab)
cm_ab = confusion_matrix(y_test, y_pred_ab)

print(f'Accuracy: {accuracy_ab:.4f}')
print(f'Classification Report:\n{report_ab}')
print(f'Confusion Matrix:\n{cm_ab}')

