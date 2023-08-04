#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)


# ### K Nearest Neighbours Prediction Model 

# In[8]:


# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

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

# Hyperparameter Tuning for K-Nearest Neighbors (KNN)
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}  # Specify the range of k values to try
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print('---- K-Nearest Neighbors (KNN) Hyperparameter Tuning ----')
print('Best k:', grid_search.best_params_['n_neighbors'])
print('Best accuracy:', grid_search.best_score_)

# Evaluate the model with the best k on the test set
best_k = grid_search.best_params_['n_neighbors']
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_knn_best = knn_best.predict(X_test)
accuracy_knn_best = accuracy_score(y_test, y_pred_knn_best)
report_knn_best = classification_report(y_test, y_pred_knn_best)
cm_knn_best = confusion_matrix(y_test, y_pred_knn_best)
print(f'Accuracy with best k: {accuracy_knn_best:.4f}')
print(f'Classification Report with best k:\n{report_knn_best}')
print(f'Confusion Matrix with best k:\n{cm_knn_best}')


# #### K Nearest Neighbours - 9 best to use 

# ### K Nearest Neighbours with Select K Best

# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

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

# Range of k_best_features to try
k_best_features_range = [5, 10, 15, 20, 25]

for k_best_features in k_best_features_range:
    # Perform Feature Selection with SelectKBest and f_classif
    selector = SelectKBest(score_func=f_classif, k=k_best_features)
    X_selected = selector.fit_transform(X_scaled, y)

    # Split the data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # Hyperparameter Tuning for K-Nearest Neighbors (KNN)
    param_grid = {'n_neighbors': [9]}  # Set k=9 for KNN
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print(f'---- K-Nearest Neighbors (KNN) with SelectKBest (k={k_best_features}) ----')
    print('Best k:', grid_search.best_params_['n_neighbors'])
    print('Best accuracy:', grid_search.best_score_)

    # Evaluate the model with the best k on the test set
    best_k = grid_search.best_params_['n_neighbors']
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train, y_train)
    y_pred_knn_best = knn_best.predict(X_test)
    accuracy_knn_best = accuracy_score(y_test, y_pred_knn_best)
    report_knn_best = classification_report(y_test, y_pred_knn_best)
    cm_knn_best = confusion_matrix(y_test, y_pred_knn_best)
    print(f'Accuracy with best k: {accuracy_knn_best:.4f}')
    print(f'Classification Report with best k:\n{report_knn_best}')
    print(f'Confusion Matrix with best k:\n{cm_knn_best}')
    print('----------------------------------------------')


# ### Select K Best - 25

# ### Optimal K Nearest Neighbours + Select K Best + Class Weight Balanced

# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

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

# Perform Feature Selection with SelectKBest and f_classif
k_best_features = 25  # You can adjust the number of features to select based on your preference
selector = SelectKBest(score_func=f_classif, k=k_best_features)
X_selected = selector.fit_transform(X_scaled, y)

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Hyperparameter Tuning for K-Nearest Neighbors (KNN)
param_grid = {'n_neighbors': [9]}  # Set k=9 for KNN
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print('---- K-Nearest Neighbors (KNN) with SelectKBest (k=25) ----')
print('Best k:', grid_search.best_params_['n_neighbors'])
print('Best accuracy:', grid_search.best_score_)

# Evaluate the model with the best k and class weights on the test set
best_k = grid_search.best_params_['n_neighbors']
knn_best = KNeighborsClassifier(n_neighbors=best_k, weights='distance')  # Use 'weights=distance' for class weights
knn_best.fit(X_train, y_train)
y_pred_knn_best = knn_best.predict(X_test)
accuracy_knn_best = accuracy_score(y_test, y_pred_knn_best)
report_knn_best = classification_report(y_test, y_pred_knn_best)
cm_knn_best = confusion_matrix(y_test, y_pred_knn_best)
print(f'Accuracy with best k and class weights: {accuracy_knn_best:.4f}')
print(f'Classification Report with best k and class weights:\n{report_knn_best}')
print(f'Confusion Matrix with best k and class weights:\n{cm_knn_best}')

