import keras
import tensorflow as tf


model = keras.models.load_model('./Models/LSTM_BTC.h5')
dot_img_file = './Data/Plots/lstm_BTC.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
model.summary()
