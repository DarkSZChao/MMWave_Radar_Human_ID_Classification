import pickle

from tensorflow.keras.models import load_model
# import numpy as np
import tensorflow as tf
# from tensorflow import keras

from NN import OrthogonalRegularizer
from config import *
from real_time_tracking.library.utils import *

if __name__ == '__main__':
    """evaluate test dataset"""
    # load data and label
    dataset_file = 'test_dataset_12_200_filtered_timelen_30'
    with open(f'{os.path.join(DATASET_DIR, dataset_file)}', 'rb') as file:
        test_data_np, test_labels_np = pickle.load(file)

    # test_data_np = test_data_np[:, :, :, np.newaxis]  # for 2D only
    # test_data_np = test_data_np[:, :, :, :, np.newaxis]  # for 2D time only

    # load the model
    # model = load_model(os.path.join(MODEL_SAVE_DIR, 'model.h5'), custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})
    model = load_model(os.path.join(BESTMODEL_SAVE_DIR, '200_filtered_timelen_30_tnet/30.h5'), custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})

    # compile the model before the evaluation
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # when label_categorical is False
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    # evaluate the model
    model.evaluate(test_data_np, test_labels_np, verbose=1)
