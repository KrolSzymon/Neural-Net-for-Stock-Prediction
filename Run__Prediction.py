import numpy as np 
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


def transform_dataset(ds, lookback):
    x = []
    y = []
    ds = np.array(ds)
    for i in range(lookback, ds.shape[0]):
        x.append(ds[i-lookback:i])                                                      
        y.append(ds[i,0])                                                               
    x= np.array(x)                                                                             
    y= np.array(y)  
    return x, y

model = load_model('transformer.h5')
dataset = 'MCD.csv'
dataset = read_csv(dataset, header = 0).dropna()
dataset = dataset.drop(labels=['Date'], axis = 1 )
dataset = MinMaxScaler().fit_transform(dataset)
print(dataset[0:10])
x, y = transform_dataset(dataset, 5)
epochs_number = 10
model.evaluate(x, y)
history = model.predict(x)
print(history)