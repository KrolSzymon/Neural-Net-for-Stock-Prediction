import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers


def split_dataset(data, train_test_split):
    train_size = int(len(data) * train_test_split)
    train, test = data[0:train_size], data[train_size:len(data)]
    return train, test  

def transform_dataset(ds, lookback):
    x = []
    y = []
    ds = np.array(ds)
    for i in range(lookback, ds.shape[0]):
        if i+1 >= len(ds): break
        x.append(ds[i-lookback:i])  
        if(ds[i + 1] == ds[i]):
            y.append(1)
        elif(ds[i + 1] > ds[i]):
            y.append(2)
        elif(ds[i + 1] < ds[i]):
            y.append(0)                                                                 
    x= np.array(x)                                                                             
    y= np.array(y)
    print(x.shape)
    print(y.shape)  
    return x, y

def transformer_encoder(inputs, head_size, number_of_heads, ff_dimensions, dropout):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim = head_size, num_heads=number_of_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dimensions, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(imput_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout = 0, mlp_dropout = 0):
    inputs = keras.Input(shape=imput_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    x = layers.GlobalAveragePooling1D(data_format = "channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs)
# Run File. _____________________________________________________________________________________________________________________________
if __name__ == "__main__":


    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
                       
    dataset_split = 0.8
    lookback = 30
    dataset_address = './Data/Learning Data/MCD.csv'
    batch_size = 30
    dropout_rate = 0.1

    dataset = pd.read_csv(dataset_address)
    dataset = dataset.drop(labels=['Date','Open','High','Low','Adj Close','Volume'], axis = 1 )
    data, test_data = split_dataset(dataset, dataset_split)
    x_train, y_train = transform_dataset(data, lookback)
    x_test, y_test = transform_dataset(test_data, lookback)

    number_of_classes = len(np.unique(y_train))
    input_shape = x_train.shape[1:]

    model_address = "./Models/transformerClasification.h5"
    model = build_model(
        input_shape,
        head_size=256,
        num_heads=8,
        ff_dim=4,
        num_transformer_blocks=8,
        mlp_units=[8],
        mlp_dropout=0.3,
        dropout=0.01,
    )
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), keras.callbacks.ModelCheckpoint(model_address, save_best_only = True, monitor = 'val_sparse_categorical_accuracy')]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    model.evaluate(x_test, y_test, verbose=1)

