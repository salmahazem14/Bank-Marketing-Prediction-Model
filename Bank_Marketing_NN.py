# # Task 2 
# # Neural Network 
# 

# **Using Bank Marketing Dataset**
# 
# The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).
# 
# 

# **1- Import Libraries**

import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import tensorflow 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv("bank-additional-full.csv")
df.head()


# **Data is not comma seperated , it is colon seperated**

df = pd.read_csv("bank-additional-full.csv" , sep=';')
df.head()

df.info()

# **Check the label --> Supervised**

df['y']

# **Label is category --> classification**

# **Check nans**

df.isnull().sum()

# **No missing values**

# **Check duplicates**

df.duplicated().sum()

df[df.duplicated()]


# **Drop duplicates**

df = df.drop_duplicates(keep='first')

# **Verify no duplicates**

df.duplicated().sum()

# **Data correlation**

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)

# **Encode the label**

df['y'] = LabelEncoder().fit_transform(df['y'])  

df['y']

# **Encode other categorical data**

df = pd.get_dummies(df, drop_first=True)  

# **Check Distributions**

df[numeric_cols].hist(bins=30, figsize=(15,10))
plt.suptitle("Histogram of Numeric Features")
plt.show()

# **Box plots to check outliers**

plt.figure(figsize=(15,10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 7, i)  
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# **Scale Numerical Features**

scaler = RobustScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# **Split the data**

X = df.drop('y', axis=1)
y = df['y']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

# # Applying Golden Rules

# **1-Few Data -> Simple Model**

X_small = X_train.sample(frac=0.05, random_state=42)
y_small = y_train.loc[X_small.index]

model_simple = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

model_simple.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_small = model_simple.fit(
    X_small, y_small,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16
)

plt.figure(figsize=(10,6))
plt.plot(history_small.history['loss'], label='Training Loss', color='blue')
plt.plot(history_small.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Step 1: Few Data → Simple Model Loss')
plt.legend()
plt.show()

# **Training loss decreases steadily which shows that the model is learning the small subset**
# 
# 

# **Validation loss is lower than training initially but fluctuates which shows that the model cannot generalize well from so few samples**

# **2-Full Data -> Simple model**

history_full_simple = model_simple.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  
    batch_size=32
)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_full_simple.history['loss'], label='Training Loss', color='blue')
plt.plot(history_full_simple.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Step 2: Full Data → Simple Model Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_full_simple.history['accuracy'], label='Training Accuracy', color='green')
plt.plot(history_full_simple.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Step 2: Full Data → Simple Model Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# The model is basically memorized the data instead of actually learning. You can see this clearly because training accuracy keeps going up while validation accuracy just flatlines around 90.7% and won't change no matter how many epochs you run. That gap between the two curves is the problem. The fix is what comes next: make the model deeper (Step 3) then add regularization (Step 4) to force it to actually generalize.

# **3-Full data -> complex model**

model_complex = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  
    Dense(32, activation='relu'),                                    
    Dense(16, activation='relu'),                                    
    Dense(1, activation='sigmoid')                                 
])

model_complex.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_complex = model_complex.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_complex.history['loss'], label='Training Loss', color='blue')
plt.plot(history_complex.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Step 2: Full Data → Simple Model Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1,2,2)
plt.plot(history_complex.history['accuracy'], label='Training Accuracy', color='green')
plt.plot(history_complex.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Step 2: Full Data → Simple Model Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# This one is actually worse than the previous plot. The model is memorizing the training data so hard that it can't generalize well on new data as training goes on — you can see the validation loss  going up while training loss goes down, and validation accuracy slowly drifting down over time. Actions ->   need to stop training much earlier, and start regularization in the next step.

# **4-Full data -> Complex Model**

from tensorflow.keras.layers import Dense, Dropout

model_complex_reg = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),                    # drop 30% of neurons randomly
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile
model_complex_reg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history_complex_reg = model_complex_reg.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_complex_reg.history['loss'], label='Training Loss', color='blue')
plt.plot(history_complex_reg.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Step 2: Full Data → Simple Model Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_complex_reg.history['accuracy'], label='Training Accuracy', color='green')
plt.plot(history_complex_reg.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Step 2: Full Data → Simple Model Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# With regularization, the model looks healthier — the loss for training and validation drops together and stays close, showing it’s learning without overfitting too much. Training accuracy still rises a bit higher than validation, but the gap is smaller, so the model generalizes better.