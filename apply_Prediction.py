
from collections import OrderedDict

import numpy as np
from keras.models import load_model
from model_PredictionTransformer import transform_dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as pp

def predict_future(prediction, training_range, num_features, timesteps_in_future):
    prediction_list = prediction[-training_range:]
    for _ in range(timesteps_in_future):      
        transformed_list = prediction_list[-training_range:,:num_features]
        transformed_list = transformed_list.reshape((1, training_range, num_features))
        out = model.predict(transformed_list)[0]
        out = np.reshape(out,(1,1,num_features))
        transformed_list = np.concatenate((transformed_list, out),axis=1)
    prediction_list = prediction_list[-timesteps_in_future:]
    return prediction_list
    


if __name__ == "__main__":
    timesteps_in_future = 10
    model = load_model('./Models/transformer.h5')
    dataset = './Data/Learning Data/MCD.csv'
    scaler = MinMaxScaler()
    lookback = 30

    dataset = read_csv(dataset, header = 0).dropna()
    dataset = dataset.drop(labels=['Date','Open','Adj Close'], axis = 1 )

    dataset = scaler.fit_transform(dataset)

    x, y = transform_dataset(dataset, lookback)
    prediction = model.predict(x)
    num_features = len(prediction[0])

    predicted_values = predict_future(prediction, lookback, num_features, timesteps_in_future) 
    predicted_values = scaler.inverse_transform(predicted_values)
    print(predicted_values)

