from model_ClassificationTransformer import split_dataset, transform_dataset
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    validation_split = 0.8
    dropout = 0.2
    lookback = 300
    epoch_amount = 100
    batch = 100
    features = 15
    units_1 = 500
    units_2 = 250
    units_3 = 125
    optimizer = keras.optimizers.Adam(learning_rate=0.5)
    data_address = './Data/HSBC.csv'
    model_address = './Models/LSTM-HSBC-LB300.h5'
    scaler = MinMaxScaler()
    data = pd.read_csv(data_address, header = 0).dropna()
    data = data.drop(labels = ['Date'], axis=1)
    data = scaler.fit_transform(data)
    
    data, test_data = split_dataset(data,validation_split)
    x_train, y_train = transform_dataset(data, lookback) 
    x_test, y_test = transform_dataset(test_data, lookback)
    input_shape = x_train.shape[1]
    feature_shape = x_train.shape[2]


    model = Sequential()
    model.add(LSTM(units=units_1,return_sequences=True,input_shape=(input_shape, feature_shape)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units_2,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units_3))
    model.add(Dropout(dropout))
    model.add(Dense(features))
    model.compile(optimizer=optimizer,loss='mean_squared_error')
    callbacks = [keras.callbacks.ModelCheckpoint(model_address, save_best_only = True, monitor = 'val_loss')]
    #model = keras.models.load_model(model_address)
    model.fit(x_train,y_train,epochs=epoch_amount,validation_split = 0.2, batch_size=batch, callbacks = [callbacks])
    output = model.predict(x_test)
    output = scaler.inverse_transform(output)

    training_performance = model.predict(x_train)
    training_performance = pd.DataFrame(data = {'Training Predictions': training_performance[:,0], 'Training Actual': y_train})
    plt.plot(training_performance['Training Predictions'], color = 'red', label = 'Predicted')
    plt.plot(training_performance['Training Actual'], color = 'blue', label = 'Actual')
    plt.legend()
    plt.show()