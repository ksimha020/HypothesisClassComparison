#designed by Kshitij Simha R (2022A7PS0572G)
# Importing all necessary libraries.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import Sequential
from keras import regularizers

#essential variable declarations
train_percent = 0.7
validation_percent = 0.2
company_sym = 'AMZN'
look_back = 270
LSTM_firstlayernodes = 140
LSTM_firstlayerWeightRegularizer = 1e-5
scaler = StandardScaler()
path_checkpoint = 'checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,monitor='val_loss',verbose=1,save_weights_only=True,save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=1)
callback_tensorboard = TensorBoard(log_dir='./logs/',histogram_freq=0, write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,min_lr=1e-4,patience=0,verbose=1)
callbacks = [callback_early_stopping,callback_checkpoint,callback_tensorboard,callback_reduce_lr]

df = pd.read_csv('Data/' + company_sym + '.csv')
data = df['Close'].values
scaled_data = scaler.fit_transform(data.reshape(-1,1))
dataset_size = len(data)
train_index = int(dataset_size*train_percent)
validation_index = int(dataset_size*validation_percent)+train_index

train_data = scaled_data[:train_index]
x_train, y_train = [], []
for i in range(look_back, train_index-look_back):
    x_train.append(train_data[i - look_back:i, 0])
    y_train.append(train_data[i,0])

validation_data = scaled_data[train_index:validation_index]
x_validation, y_validation = [], []
for i in range(look_back, validation_index-train_index-look_back):
    x_validation.append(validation_data[i-look_back:i, 0])
    y_validation.append(validation_data[i,0])

test_data = scaled_data[validation_index:]
x_test, y_test = [], []
for i in range(look_back, dataset_size-validation_index-look_back):
    x_test.append(test_data[i-look_back:i, 0])
    y_test.append(test_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_validation, y_validation = np.array(x_validation), np.array(y_validation)
x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], 1))
validation_data = (np.expand_dims(x_validation, axis=0), np.expand_dims(y_validation, axis=0))
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=LSTM_firstlayernodes, input_shape=(x_train.shape[1], 1), kernel_regularizer=regularizers.l2(LSTM_firstlayerWeightRegularizer)))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=1, epochs=20, callbacks=callbacks, validation_data=(x_validation, y_validation))

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

result = model.evaluate(x_test, y_test)
print("loss (test-set):", result)


plt.figure()
plt.title('Test Data')
plt.plot(y_test, label='true')
plt.plot(model.predict(x_test), label='predicted')
plt.legend()
plt.show()


plt.figure()
plt.title('All data')
x_final, y_true = [], []
final_data = scaler.inverse_transform(scaled_data)
for i in range(look_back, dataset_size-look_back):
    x_final.append(scaled_data[i-look_back:i, 0])
    y_true.append(final_data[i,0])
x_final = np.array(x_final)
y_true = np.array(y_true)
x_final = np.reshape(x_final, (x_final.shape[0], x_final.shape[1], 1))
plt.plot(scaler.inverse_transform(model.predict(x_final)), label='predicted')
plt.plot(y_true, label='true')
plt.legend()
plt.show()