
from collections import OrderedDict

import numpy as np
from keras.models import load_model
from Transformer_Model_Creation import transform_dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as pp


timesteps_in_future = 10
model = load_model('model_btc.h5 ')
dataset = 'MCD.csv'
scaler = MinMaxScaler()
lookback = 30

dataset = read_csv(dataset, header = 0).dropna()
dataset = dataset.drop(labels=['Date','Open','High','Low','Adj Close','Volume'], axis = 1 )
dataset = scaler.fit_transform(dataset)
x, y = transform_dataset(dataset, lookback)

prediction = model.predict(x) 


def predict_future(transformed_dataset, training_range):
    prediction_list = prediction[-training_range:]    
    for _ in range(timesteps_in_future):
        transformed_dataset = prediction_list[-training_range:]
        transformed_dataset = transform_dataset.reshape((1, training_range, 1))
        out = model.predict(transformed_dataset)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[training_range-1:]
    return prediction_list
    


if __name__ == "__main__":
    predicted_values = predict_future(x, lookback)

