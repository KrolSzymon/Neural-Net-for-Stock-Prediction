
from collections import OrderedDict

import numpy as np
from keras.models import load_model
from Classification_Model_Creation import transform_dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as pp


timesteps_in_future = 10
model = load_model('./Models/transformerClasification.h5')
dataset = './Data/Learning Data/MCD.csv'
lookback = 30

dataset = read_csv(dataset, header = 0).dropna()
dataset = dataset.drop(labels=['Date','Open','High','Low','Adj Close','Volume'], axis = 1 )
x, y = transform_dataset(dataset, lookback)




def predict_future(transformed_dataset, training_range):
    prediction_list = transformed_dataset[-training_range:]    
    transformed_dataset = prediction_list[-training_range:]
    out = model.predict(transformed_dataset)
    return out
    


if __name__ == "__main__":
    predicted_values = predict_future(x, lookback)
    print(predicted_values.shape)

