import numpy as np 
import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
 
# Configure hyperparameters. _____________________________________________________________________________________________________________

# ADAMW optimiser hyperparameters.
learning_rate = 0.001
weight_decay = 0.001

# Transformer hyperparameters.
batch_size = 30
num_epochs = 100
transformer_layers = 8

# Data hyperparameters.
dataset = 'MCD.csv'
lookback = 5
train_test_split = 0.8

# Size of hidden dimension feature vectors.
projection_dim = 30

# Number of transformations of query-key-value matrices.
num_attention_heads = 16

# Input columnn number
mlp_feature_dim = [6]


# Split dataset into train and test ______________________________________________________________________________________________________
def split_dataset(data, train_test_split):
    train_size = int(len(data) * train_test_split)
    train, test = data[0:train_size], data[train_size:len(data)]
    return train, test  

# Load the dataset. ______________________________________________________________________________________________________________________
def transform_dataset(ds, lookback):
    x = []
    y = []
    ds = np.array(ds)
    print(ds[0:10])
    for i in range(lookback, ds.shape[0]):
        x.append(ds[i-lookback:i])                                                      
        y.append(ds[i,0])                                                               
    x= np.array(x)                                                                             
    y= np.array(y)  
    return x, y

# Define model architecture. _____________________________________________________________________________________________________________
def transformer(inputs, head_size, num_heads, dropout = 0):
    input = layers.Input(shape = inputs.shape[1:])
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(input)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            dropout=dropout,
        )(x1, x1)
        x2 = layers.Add()([attention_output, input])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units = mlp_feature_dim, dropout_rate = dropout)
        
        res = layers.Add()([x3, x2])
    
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = mlp(x, hidden_units = mlp_feature_dim, dropout_rate = dropout)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs = input, outputs = outputs)
    return keras.Model(input, outputs)

# Define MLP layer.
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation = tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x    

#Build model______________________________________________________________________________________________________________________________
def run_vit_prediction(x_tr, y_tr, x_te, y_te, projection_dim, num_attention_heads, batch_size, num_epochs):
    model = transformer(
        x_tr,
        head_size = projection_dim,
        num_heads = num_attention_heads,
    )
    model.compile(
        loss = keras.losses.MeanSquaredError(),
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
    )
    model.summary()
    callbacks = [keras.callbacks.ModelCheckpoint("transformer.h5", save_best_only = True, monitor = 'val_loss')]
    history = model.fit(
        x_tr,
        y_tr,
        validation_split = 0.2,
        epochs = num_epochs,
        batch_size = batch_size,
        callbacks = callbacks
    )
    model.evaluate(x_te, y_te, verbose = 1)
    return history


def plot_results(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()
    
# Run File. _____________________________________________________________________________________________________________________________
if __name__ == "__main__":
# Check for GPU's.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)           
# Load dataset.
    ds = pd.read_csv(dataset, header = 0).dropna()
    ds = ds.drop(labels=['Date'], axis = 1 )
# OPTIONAL FEATURE - Research needed to determine if this is a good idea. 
    ds = MinMaxScaler().fit_transform(ds)
    
    train, test = split_dataset(ds, train_test_split)  
    train_x, train_y = transform_dataset(train, lookback)
    test_x, test_y = transform_dataset(test, lookback)
    history = run_vit_prediction(train_x, train_y, test_x, test_y, projection_dim, num_attention_heads, batch_size, num_epochs)
    plot_results(history)