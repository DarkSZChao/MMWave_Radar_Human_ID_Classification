import math
import os
import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical

from real_time_tracking.library.utils import random_split

# enable GPU for calculation
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 0 for GPU, -1 for CPU
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))


def scheduler(epoch, lr):
    if epoch < 5:
        lr = 0.0002
    elif epoch == 50:
        lr = 0.0002
    else:
        # lr = lr * 0.95
        lr = lr * math.exp(-lr * 100)
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)


def _dense_bn_relu(filters, layer):
    layer = Dense(filters)(layer)
    layer = BatchNormalization()(layer)
    return Activation("relu")(layer)


def _conv1d_bn_relu(filter_No, kernel_size, layer):
    layer = Conv1D(filter_No, kernel_size, strides=1, padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    return layer


def model_structure_MLP(input_shape, output_shape):
    _model_input = Input(shape=input_shape)
    _model = Flatten()(_model_input)
    _model = _dense_bn_relu(2048, _model)
    _model = Dropout(rate=0.1)(_model)
    _model = _dense_bn_relu(1024, _model)
    _model = Dropout(rate=0.1)(_model)
    _model = _dense_bn_relu(512, _model)
    _model = Dropout(rate=0.1)(_model)
    _model = _dense_bn_relu(256, _model)
    _model = Dropout(rate=0.1)(_model)
    _model = _dense_bn_relu(128, _model)
    _model = Dropout(rate=0.1)(_model)
    _model = Dense(output_shape, activation='softmax')(_model)
    return Model(_model_input, _model)


def model_structure_Conv1D(input_shape, output_shape):
    _model_input = Input(shape=input_shape)
    _model = _conv1d_bn_relu(512, 3, _model_input)
    _model = MaxPooling1D(pool_size=2)(_model)

    _model = _conv1d_bn_relu(512, 3, _model)
    _model = MaxPooling1D(pool_size=2)(_model)

    _model = _conv1d_bn_relu(256, 3, _model)
    _model = MaxPooling1D(pool_size=2)(_model)

    _model = _conv1d_bn_relu(128, 3, _model)
    _model = MaxPooling1D(pool_size=2)(_model)

    _model = _conv1d_bn_relu(128, 3, _model)
    _model = MaxPooling1D(pool_size=2)(_model)

    _model = Flatten()(_model)
    _model = _dense_bn_relu(128, _model)
    _model = Dropout(rate=0.1)(_model)
    _model = _dense_bn_relu(64, _model)
    _model = Dropout(rate=0.1)(_model)
    _model = Dense(output_shape, activation='softmax')(_model)
    return Model(_model_input, _model)


def model_structure_Conv2D(input_shape, output_shape):
    _model_input = Input(shape=input_shape)
    _model = Conv2D(480, (3, 3), padding='same', activation='relu')(_model_input)
    _model = MaxPooling2D(pool_size=(2, 1))(_model)
    _model = Conv2D(320, (3, 3), padding='same', activation='relu')(_model)
    _model = MaxPooling2D(pool_size=(2, 1))(_model)
    _model = Conv2D(240, (3, 3), padding='same', activation='relu')(_model)
    _model = MaxPooling2D(pool_size=(2, 1))(_model)
    _model = Conv2D(120, (3, 3), padding='same', activation='relu')(_model)
    _model = MaxPooling2D(pool_size=(2, 1))(_model)
    _model = Flatten()(_model)
    _model = Dense(128)(_model)
    _model = Dense(output_shape, activation='softmax')(_model)
    return Model(_model_input, _model)


def load_data(label_categorical=False):
    """
    Faust point cloud data only
    """
    with open('../Data/FAUSTSim/faust_ori_pointcloud_256_10000_dis', 'rb') as file:
        datapoints_np, labels_np = pickle.load(file)

    # define dataset params
    people = 10
    train_ratio = 0.8
    train_data_np = np.empty([0] + list(datapoints_np.shape[1:]))
    train_labels_np = np.empty([0] + list(labels_np.shape[1:]))
    val_data_np = np.empty([0] + list(datapoints_np.shape[1:]))
    val_labels_np = np.empty([0] + list(labels_np.shape[1:]))
    for person in range(people):
        # random split whole dataset into train and valid in posture dimension
        idx = np.squeeze(np.argwhere(labels_np[:, 0] == person))
        train_idx, val_idx = random_split(idx, split_ratio=train_ratio)
        train_data_np = np.concatenate([train_data_np, datapoints_np[train_idx]])
        train_labels_np = np.concatenate([train_labels_np, labels_np[train_idx]])
        val_data_np = np.concatenate([val_data_np, datapoints_np[val_idx]])
        val_labels_np = np.concatenate([val_labels_np, labels_np[val_idx]])

    if label_categorical:
        return train_data_np, to_categorical(train_labels_np), val_data_np, to_categorical(val_labels_np)
    else:
        return train_data_np, train_labels_np, val_data_np, val_labels_np


if __name__ == '__main__':
    # load data from stored dataset
    train_data_np, train_labels_np, val_data_np, val_labels_np = load_data(label_categorical=True)

    model = model_structure_Conv1D(train_data_np.shape[1:], 10)
    # train_data_np = train_data_np[:, :, :, np.newaxis]
    # val_data_np = val_data_np[:, :, :, np.newaxis]
    # model = model_structure_Conv2D(train_data_np.shape[1:], 10)
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_data_np, train_labels_np,
                        validation_data=(val_data_np, val_labels_np),
                        shuffle=True,
                        batch_size=128,
                        callbacks=[lr_callback],
                        epochs=300,
                        verbose=1)

    print(f"Epoch {np.argmax(history.history['val_accuracy']) + 1} with maximum val_accuracy: {history.history['val_accuracy'][np.argmax(history.history['val_accuracy'])] * 100:.2f}%")

    with open('faust_ori_pointcloud_256_10000_dis_history', 'wb') as f:
        pickle.dump(history.history, f)

    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()
