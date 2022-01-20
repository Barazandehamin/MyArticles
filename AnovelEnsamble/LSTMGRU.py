import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers



data= pd.read_csv("source_price.csv")
length_data=len(data)
#print(data.describe())
data['reuters_mean_compound'].plot()
data['wsj_mean_compound'].plot()
data['cnbc_mean_compound'].plot()
data['fortune_mean_compound'].plot()

y=data.close
x=data.drop('close',axis=1)
x=x.drop('date',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")


# variables
n=50 #of neurons
inp_steps=1 #input time esteps
input_dim =4
drop_out=0.2
epochs = 10
batch_size=4

#LSTM Model
model = Sequential()
model.add(LSTM(n, input_shape=(inp_steps, input_dim), return_sequences = True))
model.add(Dropout(drop_out))
model.add(LSTM(n,return_sequences = True))
model.add(Dropout(drop_out))
model.add(LSTM(n,return_sequences = True))
model.add(Dropout(drop_out))
model.add(LSTM(n,return_sequences = True))
model.add(Dropout(drop_out))
# Compile model
model.compile(loss='mean_squared_error',
                optimizer='adam')
# Fit the model
model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
#model.summary()
#model.predict(x_test)


#GRU Model
modelg=Sequential()
modelg.add(layers.GRU(n, input_shape=(inp_steps, input_dim), return_sequences = True))
modelg.add(Dropout(drop_out))

modelg.add(layers.GRU(50, return_sequences=True))
modelg.add(Dropout(drop_out))

modelg.add(layers.GRU(50, return_sequences=True))
modelg.add(Dropout(drop_out))

modelg.add(layers.GRU(50, return_sequences=True))
modelg.add(Dropout(drop_out))

modelg.compile(loss='mean_squared_error',optimizer='adam')
modelg.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)

modelg.summary()

#Meta Learner

modelm = Sequential()
modelm.add(Dense(12, input_dim=8, activation='relu'))

