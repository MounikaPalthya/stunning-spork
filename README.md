# stunning-spork
RAINFALL DETECTION Machine Learning model
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

from google.colab import files
import zipfile
uploaded = files.upload()
# Replace the filename with the name of your zip file
zip_filename = 'archive.zip'

# Extract the zip file
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall()

# Load the dataset
data = pd.read_csv('weatherAUS.csv')


# Filter relevant columns and remove missing values
data = data[['Date', 'Location', 'Rainfall']]
data = data.dropna()

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Rainfall']].values)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]

# Define the training data and labels
X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, :])
    y_train.append(train_data[i, 0])

# Reshape the training data
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Define the testing data and labels
X_test = []
y_test = []
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, :])
    y_test.append(test_data[i, 0])

# Reshape the testing data
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions on the testing data
predictions = model.predict(X_test)

# Evaluate the model
mse = np.mean((predictions - y_test)**2)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)
