import numpy as np
from keras.models import load_model
from model_PredictionTransformer2 import transform_dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy import newaxis
import matplotlib.pyplot as plt

def predict_future(prediction, training_range, num_features, timesteps_in_future):
    transformed_list= prediction[-training_range:]

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
    model = load_model('./Models/transformer.h5')
    dataset = './Data/BTC.csv'
    scaler = MinMaxScaler()
    lookback = 30
    plt.style.use('default')

    dataset = read_csv(dataset, header = 0).dropna()
    dataset_no_date = dataset.drop(labels=['Date'], axis = 1 )
    dataset_no_date = dataset_no_date.astype('float32')
    
   

    num_features = len(dataset_no_date.columns)
    dataset_scaled = scaler.fit_transform(dataset_no_date)
    dataset_scaled = np.array(dataset_scaled)
    
    predicted_values = predict_future(dataset_scaled, lookback, num_features, timesteps_in_future) 
    predicted_values = scaler.inverse_transform(predicted_values)
    
    dataset.index = pd.DatetimeIndex(dataset['Date'])

    predicted_values = pd.DataFrame(data = predicted_values, columns = dataset_no_date.columns)

    last_date = dataset.tail(1).index.strftime("%Y-%m-%d")[0]
    last_date = pd.to_datetime(last_date)

    first_date = last_date + pd.DateOffset(days = 1)
    predicted_values.index = pd.DatetimeIndex(pd.date_range(first_date, periods = timesteps_in_future, freq = 'D'))

    plt.figure(figsize = (20,10), dpi = 150)

    dataset['Open'].plot(label = 'Open' , color = 'blue')
    predicted_values['Open'].plot(label = 'Open - Prediction', color = 'red')


    plt.legend()
    plt.title('BTC and S&P 500 Open Price Prediction')
    plt.show()


  

