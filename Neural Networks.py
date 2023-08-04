#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Replace 'path/to/income_classification.csv' with the actual path to your file
file_path = 'income_evaluation.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)


# ### Neural Network Prediction Model 

# In[3]:


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

# Neural Network Model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test)
y_pred_nn = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to class labels (0 or 1)
accuracy_nn = accuracy_score(y_test, y_pred_nn)
report_nn = classification_report(y_test, y_pred_nn)
cm_nn = confusion_matrix(y_test, y_pred_nn)

print('---- Neural Network Model ----')
print(f'Accuracy: {accuracy_nn:.4f}')
print(f'Classification Report:\n{report_nn}')
print(f'Confusion Matrix:\n{cm_nn}')


# ### Neural Network Prediction Model - Increase the number of layers and neurons 

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout

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

# Neural Network with Increased Layers and Nodes
print('---- Neural Network ----')
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))  # Increase the number of neurons to 128
model.add(Dense(64, activation='relu'))  # Add another hidden layer with 64 neurons
model.add(Dense(32, activation='relu'))  # Add another hidden layer with 32 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test)
y_pred_nn = (y_pred_prob > 0.5).astype(int).flatten()
accuracy_nn = accuracy_score(y_test, y_pred_nn)
report_nn = classification_report(y_test, y_pred_nn)
cm_nn = confusion_matrix(y_test, y_pred_nn)
print(f'Accuracy: {accuracy_nn:.4f}')
print(f'Classification Report:\n{report_nn}')
print(f'Confusion Matrix:\n{cm_nn}')


# ### Neural Network Prediction Model with Regularisation  

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

# Neural Network Model with Dropout Regularization
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Add dropout layer with 20% dropout rate
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))  # Add dropout layer with 20% dropout rate
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test)
y_pred_nn = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to class labels (0 or 1)
accuracy_nn = accuracy_score(y_test, y_pred_nn)
report_nn = classification_report(y_test, y_pred_nn)
cm_nn = confusion_matrix(y_test, y_pred_nn)

print('---- Neural Network Model with Dropout Regularization ----')
print(f'Accuracy: {accuracy_nn:.4f}')
print(f'Classification Report:\n{report_nn}')
print(f'Confusion Matrix:\n{cm_nn}')


# ### Neural Network Prediction Model with Batch Normalisation

# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam

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

# Neural Network Model with Batch Normalization
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())  # Add batch normalization layer after the first hidden layer
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())  # Add batch normalization layer after the second hidden layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test)
y_pred_nn = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to class labels (0 or 1)
accuracy_nn = accuracy_score(y_test, y_pred_nn)
report_nn = classification_report(y_test, y_pred_nn)
cm_nn = confusion_matrix(y_test, y_pred_nn)

print('---- Neural Network Model with Batch Normalization ----')
print(f'Accuracy: {accuracy_nn:.4f}')
print(f'Classification Report:\n{report_nn}')
print(f'Confusion Matrix:\n{cm_nn}')

