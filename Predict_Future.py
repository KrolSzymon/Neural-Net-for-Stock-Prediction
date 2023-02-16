
from collections import OrderedDict

import numpy as np
from keras.models import load_model
from Run__Prediction import transform_dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

timesteps_in_future = 10
predictions = np.array([])
model = load_model('transformer.h5')
dataset = 'MCD.csv'
dataset = read_csv(dataset, header = 0).dropna()
dataset = dataset.drop(labels=['Date'], axis = 1 )
dataset = MinMaxScaler().fit_transform(dataset)
print(dataset[0:10])
x, y = transform_dataset(dataset, 5)
last_timestep = x[-1]

for i in range(timesteps_in_future):
    prediction = model.predict(np.array([last_timestep]))
    last_timestep = np.concatenate((last_timestep[1:], prediction[0]))
    prediction_array = np.concatenate(prediction_array, prediction[0])
prediction_array = MinMaxScaler.inverse_transform(prediction_array)
print(prediction_array)
