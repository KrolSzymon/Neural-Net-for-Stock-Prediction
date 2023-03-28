from model_PredictionTransformer import split_dataset, transform_dataset
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":

    data_address = './Data/Learning Data/MCD.csv'
    model_address = './Models/LSTM.h5'
    validation_split = 0.8
    lookback = 30
    epoch_amount = 1
    batch = 30
    features = 4

    scaler = MinMaxScaler()
  
    data = pd.read_csv(data_address, header = 0).dropna()
    data = data.drop(labels = ['Date', 'Open','Adj Close'], axis=1)
    data = scaler.fit_transform(data)
    
    data, test_data = split_dataset(data,validation_split)
    x_train, y_train = transform_dataset(data, lookback) 
    x_test, y_test = transform_dataset(test_data, lookback)
    input_shape = x_train.shape[1]
    feature_shape = x_train.shape[2]


    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(input_shape, feature_shape)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(features))
    model.compile(optimizer='adam',loss='mean_squared_error')
    callbacks = [keras.callbacks.ModelCheckpoint(model_address, save_best_only = True, monitor = 'val_loss')]
    model.fit(x_train,y_train,epochs=epoch_amount,validation_split = 0.2, batch_size=batch, callbacks = callbacks)
    output = model.predict(x_test)
    output = scaler.inverse_transform(output)
    print(output)
