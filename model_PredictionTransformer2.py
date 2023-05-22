import keras
from keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from model_ClassificationTransformer import split_dataset
 

def transform_dataset(ds, lookback):
    x = []
    y = []
    ds = np.array(ds)
    for i in range(lookback, ds.shape[0]):
        x.append(ds[i-lookback:i])                                                      
        y.append(ds[i,0])                                                               
    x= np.array(x).astype('float32')                                                                             
    y= np.array(y).astype('float32')
    return x, y
def transformer(inputs, number_heads, size_heads, feature_dimensions, dropout):

    x = layers.LayerNormalization(epsilon = 1e-6)(inputs)
    attention = layers.MultiHeadAttention(num_heads = number_heads, key_dim = size_heads, dropout = dropout)(x, x)
    x = layers.Dropout(dropout)(attention)

    res = x + inputs

    x = layers.LayerNormalization(epsilon = 1e-6)(res)
    x = layers.Conv1D(filters = feature_dimensions, kernel_size = 1, activation = 'relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters = feature_dimensions, kernel_size = 1)(x)
    return x + res

def build_model(input_shape, size_heads, number_heads, feature_dimensions, number_blocks, perceptron_units, dropout, mlp_dropout):
    inputs = keras.Input(shape = input_shape)
    x = inputs
    for _ in range(number_blocks):
        x = transformer(x, number_heads, size_heads, feature_dimensions, dropout)

    x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in perceptron_units:
        x = layers.Dense(dim, activation = 'relu')(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(feature_dimensions)(x)
    return keras.Model(inputs, outputs)

def plot_results(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    model_address = './Models/transformer_BTC.h5'
    dataset = './Data/BTC.csv'
    head_size = 128
    number_heads = 4
    feature_dimensions = 15
    number_blocks = 1
    perceptron_units = [4]
    dropout = 0.1
    mlp_dropout = 0.3
    lookback = 20
    train_test_split = 0.9
    number_epochs = 200
    batch_size = 20
    validation_split = 0.15
    scaler = MinMaxScaler()

    dataset = pd.read_csv(dataset).dropna()
    dataset = dataset.drop('Date', axis = 1)
    dataset = MinMaxScaler().fit_transform(dataset)
    train, test = split_dataset(dataset, train_test_split)  
    train_x, train_y = transform_dataset(train, lookback)
    test_x, test_y = transform_dataset(test, lookback)
    inpout_shape = train_x.shape[1:]

    model = build_model(inpout_shape, head_size, number_heads, feature_dimensions, number_blocks, perceptron_units, dropout, mlp_dropout)
    #model = keras.models.load_model("./Models/transformer_BTC(pretty good).h5")
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_squared_error')
    callbacks = [keras.callbacks.ModelCheckpoint(model_address, save_best_only = True, monitor = 'val_loss')]
    history = model.fit(train_x, train_y, epochs = number_epochs, batch_size = batch_size, validation_split = validation_split, callbacks = callbacks)
    plot_results(history)

    training_performance = model.predict(train_x)
    training_performance = pd.DataFrame(data = {'Training Predictions': training_performance[:,0], 'Training Actual': train_y})
    plt.plot(training_performance['Training Predictions'], color = 'red', label = 'Predicted')
    plt.plot(training_performance['Training Actual'], color = 'blue', label = 'Actual')
    plt.title('Training Performance')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.tick_params(
        axis='both',    
        which='both',      
        bottom=False,     
        left=False,
        labelbottom=False,
        labelleft=False,
        )
    plt.legend()
    plt.show()