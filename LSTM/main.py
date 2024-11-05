import numpy as np 
import pandas as pd
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


def transform_data(df):
    # Exclude the first column ('日期') and get column names
    data_index = df.columns[1:]  
   
    # Flatten the data
    flatten_data = df[data_index].values.reshape(-1)
    
    # Convert to string and clean data
    str_data = "<SEP>".join(flatten_data.astype('str'))
    filter_data = str_data.replace(',',"").replace('X',"")

    # Split back to array
    x_data = filter_data.split("<SEP>")
    
    return x_data

x = []
for path in os.listdir('Stock'):
    file_path = f'Stock/{path}'
    print(f"Loading file: {file_path}")
    df = pd.read_csv(file_path)
    data = transform_data(df)
    x.extend(data)

# Convert to NumPy array and reshape
x = np.array(x).astype('float')
x = x.reshape(-1, len(df.columns[1:]))

# Extract target variable (assuming '收盤價' is at index 5)
y = x[:,5]

# Split data into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=22, shuffle=False)

# Fit scalers on training data
sc_x = MinMaxScaler()
x_train_scaled = sc_x.fit_transform(x_train)
x_valid_scaled = sc_x.transform(x_valid)

sc_y = MinMaxScaler()
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).reshape(-1)
y_valid_scaled = sc_y.transform(y_valid.reshape(-1,1)).reshape(-1)

# Define function to split data into sequences
def split_data(datas, labels, split_num=10):
    max_len = len(datas)
    x, y = [], []
    for i in range(max_len - split_num):
        x.append(datas[i: i + split_num])
        y.append(labels[i + split_num])
    
    return np.array(x), np.array(y)

# Apply sequence splitting
x_train_seq, y_train_seq = split_data(x_train_scaled, y_train_scaled)
x_valid_seq, y_valid_seq = split_data(x_valid_scaled, y_valid_scaled)

print(f"x_train_seq shape: {x_train_seq.shape}")
print(f"y_train_seq shape: {y_train_seq.shape}")
print(f"x_valid_seq shape: {x_valid_seq.shape}")
print(f"y_valid_seq shape: {y_valid_seq.shape}")

# Build and compile the model
model= Sequential()
model.add(Bidirectional(LSTM(128, input_shape=(x_train_seq.shape[1], x_train_seq.shape[2]), return_sequences=True, activation='relu')))
model.add(Bidirectional(LSTM(64, return_sequences=False, activation='relu')))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(x_train_seq, y_train_seq,
                    batch_size=64,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_valid_seq, y_valid_seq))

# Make predictions on validation data
y_pred = model.predict(x_valid_seq)

# Inverse transform predictions and actual values
y_pred_actual = sc_y.inverse_transform(y_pred)
y_valid_actual = sc_y.inverse_transform(y_valid_seq.reshape(-1,1))

# Predict the next day's closing price
x_total_scaled = sc_x.transform(x)
last_sequence = x_total_scaled[-10:]  # Get the last sequence
next_day_pred = model.predict(np.expand_dims(last_sequence, axis=0))
next_day_pred_actual = sc_y.inverse_transform(next_day_pred)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_valid_actual, label='Actual Closing Price')
plt.plot(y_pred_actual, label='Predicted Closing Price')

# Plot the next day's prediction
plt.axvline(x=len(y_valid_actual), color='r', linestyle='--', label='Prediction Start')
plt.plot(len(y_valid_actual), next_day_pred_actual[0][0], 'ro', label='Next Day Prediction')

plt.title('Actual vs Predicted Closing Prices')
plt.ylabel('Closing Price')
plt.xlabel('Days')
plt.legend(loc='upper left')
plt.show()

print(f"Next day's predicted closing price: {next_day_pred_actual[0][0]}")
