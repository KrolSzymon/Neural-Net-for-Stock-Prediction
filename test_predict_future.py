import numpy as np
from numpy import random
from model_PredictionTransformer import transform_dataset

prediction_list = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
x, y = transform_dataset(prediction_list, 2)
print(x,y)