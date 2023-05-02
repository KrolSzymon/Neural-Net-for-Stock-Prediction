from model_PredictionTransformer2 import split_dataset, transform_dataset
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Dense, TimeDistributed, ConvLSTM2D
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

validation_split = 0.8
dropout = 0.2
lookback = 300
epoch_amount = 100
batch = 100
features = 15
units_1 = 150
units_2 = 100
units_3 = 50
units_4 = 25
optimizer = keras.optimizers.Adam(learning_rate=0.5)
scaler = MinMaxScaler()
if __name__ == "__main__":

    data_address = './Data/Learning Data/snp_btc_fullscope_daily.csv'
    model_address = './Models/CNN_LSTM.h5'

  
    data = pd.read_csv(data_address, header = 0).dropna()
    data = data.drop(labels = ['Date'], axis=1)
    data = scaler.fit_transform(data)
    
    data, test_data = split_dataset(data,validation_split)
    x_train, y_train = transform_dataset(data, lookback) 
    x_test, y_test = transform_dataset(test_data, lookback)
    input_shape = x_train.shape[1]
    feature_shape = x_train.shape[2]


    model = Sequential()

    model.add(Conv1D(filters=feature_shape, kernel_size=(lookback), activation='relu', input_shape=( input_shape, feature_shape)))
    model.add(TimeDistributed(Dense(units=units_1, activation='relu')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=units_3,return_sequences=True))
    model.add(LSTM(units=units_4))
    model.add(Dense(features))
    model.compile(optimizer=optimizer,loss='mean_squared_error')
    callbacks = [keras.callbacks.ModelCheckpoint(model_address, save_best_only = True, monitor = 'val_loss')]
    model.fit(x_train,y_train,epochs=epoch_amount,validation_split = 0.2, batch_size=batch, callbacks = [callbacks])
    output = model.predict(x_test)
    model = keras.models.load_model(model_address)
    training_performance = model.predict(x_train)
    training_performance = pd.DataFrame(data = {'Training Predictions': training_performance[:,0], 'Training Actual': y_train})
    plt.plot(training_performance['Training Predictions'], color = 'red', label = 'Predicted')
    plt.plot(training_performance['Training Actual'], color = 'blue', label = 'Actual')
    plt.legend()
    plt.show()