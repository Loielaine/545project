import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Input, TimeDistributed

# read data
dir = '/Users/Loielaine/Desktop/umich-2019/EECS545/project/code/545project/'
train = np.loadtxt(dir+'train_sample5.csv',delimiter=',')
test = np.loadtxt(dir+'test_sample5.csv',delimiter=',')

X_train = train[:,:-3]
y_train = train[:,-3:]
X_test = test[:,:-3]
y_test = test[:,-3:]

# scaling
def MinMaxScaling(X):
    scaler =preprocessing.MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X

X_train = MinMaxScaling(X_train)
X_test = MinMaxScaling(X_test)

# reshape feature matrix
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_train.shape

# reshape feature matrix
X_test = X_test.reshape(X_test.shape[0], 1,X_test.shape[1])
X_test.shape

y_train_new = y_train.reshape(y_train.shape[0],1,y_train.shape[1])

# train autoencoder
inputs_ae = Input(shape=(1,40))
encoded_ae = LSTM(128, return_sequences=True, dropout=0.3)(inputs_ae, training=True)
decoded_ae = LSTM(64, return_sequences=True, dropout=0.3)(encoded_ae, training=True)
out_ae = TimeDistributed(Dense(3))(decoded_ae)
sequence_autoencoder = Model(inputs_ae, out_ae)
sequence_autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
sequence_autoencoder.fit(X_train, y_train_new, batch_size=24, epochs=50, verbose=2, shuffle=True)

# encode X
encoder = Model(inputs_ae, encoded_ae)
X_train_encode = encoder.predict(X_train)
X_train_new = np.concatenate([X_train, X_train_encode],axis=2)
X_test_encode = encoder.predict(X_test)
X_test_new = np.concatenate([X_test, X_test_encode],axis=2)

# LSTM model
model6.add(LSTM(input_shape=(1,168),units=100,return_sequences=True))
model6.add(Dropout(0.3))
model6.add(LSTM(input_shape=(None,100),units=100,return_sequences=True))
model6.add(Dropout(0.3))
model6.add(LSTM(100,return_sequences=False))
model6.add(Dropout(0.3))
model6.add(Dense(3, activation='relu'))
model6.compile(loss="mse", optimizer="rmsprop")
print(model6.summary())
history6 = model6.fit(X_train_new,y_train,batch_size=24,epochs=100,validation_split=0.1)

# plot history
plt.plot(history6.history['loss'], label='train')
plt.plot(history6.history['val_loss'], label='valid')
plt.legend()
plt.show()

# evaluate model
y_pred0 = model6.predict(X_test_new)[:,0]
mse0 = mean_squared_error(y_test[:,0],y_pred0)
np.sqrt(mse0) #7.72

y_pred1 = model6.predict(X_test_new)[:,1]
mse1 = mean_squared_error(y_test[:,1],y_pred1)
np.sqrt(mse1) #12.75

y_pred2 = model6.predict(X_test_new)[:,2]
mse2 = mean_squared_error(y_test[:,2],y_pred2)
np.sqrt(mse2) #22.31