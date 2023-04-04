
from collections import OrderedDict

import numpy as np
from keras.models import load_model
from model_PredictionTransformer import transform_dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as pp
from numpy import newaxis

def predict_future(prediction, training_range, num_features, timesteps_in_future):
    transformed_list= prediction[-training_range:]
    print(prediction[-1])
    for _ in range(timesteps_in_future):      
        transformed_list = transformed_list.reshape((1, training_range, num_features))
        out = model.predict(transformed_list)
        out = np.reshape(out,(1,1,num_features))
        transformed_list = transformed_list[:,-training_range:,:]
        transformed_list = np.concatenate((transformed_list, out),axis=1)
        transformed_list = np.delete(transformed_list,(0),1)
    prediction_list = transformed_list[0,-timesteps_in_future:,:]
    return prediction_list
    


if __name__ == "__main__":
    timesteps_in_future = 10
    model = load_model('./Models/LSTM.h5')
    dataset = './Data/Learning Data/snp_btc_fullscope_daily.csv'
    scaler = MinMaxScaler()
    lookback = 30

    dataset = read_csv(dataset, header = 0).dropna()
    dataset = dataset.drop(labels=['Date'], axis = 1 )

    dataset = scaler.fit_transform(dataset)

    x, y = transform_dataset(dataset, lookback)
    prediction = model.predict(x)
    print(prediction[-1])
    num_features = len(prediction[0])

    predicted_values = predict_future(prediction, lookback, num_features, timesteps_in_future) 
    predicted_values = scaler.inverse_transform(predicted_values)
    print(predicted_values)

