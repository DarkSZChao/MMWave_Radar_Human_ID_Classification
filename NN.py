import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import *

from config import *

# allocate GPU memory dynamically
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def scheduler(epoch, lr):
    if epoch < 10:
        lr = 0.0001
    # elif epoch == 50:
    #     lr = 0.002
    else:
        # lr = lr * 0.95
        lr = lr * math.exp(-lr * 5)
    return lr


class TF_callbacks:
    def __init__(self):
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CP_DIR, 'ep{epoch:03d}_trainacc_{accuracy:.3f}_valloss{val_loss:.2f}_valacc{val_accuracy:.3f}.h5'),
                                                              verbose=1,
                                                              save_weights_only=True)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """
    used in tnet block for pointnet structure
    """

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
        pass

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return {'num_features': self.num_features,
                'l2reg'       : self.l2reg}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNN(TF_callbacks):
    """
    CNN blocks pre-defined
    """

    @staticmethod
    def _conv1d_bn_relu(filter_No, kernel_size, layer, time_related_enable=False):
        if not time_related_enable:
            layer = Conv1D(filter_No, kernel_size, strides=1, padding='same')(layer)
            layer = BatchNormalization(momentum=0.0)(layer)
            return Activation("relu")(layer)
        else:
            layer = TimeDistributed(Conv1D(filter_No, kernel_size, strides=1, padding='same'))(layer)
            layer = TimeDistributed(BatchNormalization(momentum=0.0))(layer)
            return TimeDistributed(Activation("relu"))(layer)

    @staticmethod
    def _conv2d_bn_relu(filter_No, kernel_size, layer):
        layer = Conv2D(filter_No, kernel_size, strides=1, padding='same')(layer)
        layer = BatchNormalization(momentum=0.0)(layer)
        return Activation("relu")(layer)

    @staticmethod
    def _dense_bn_relu(filters, layer):
        layer = Dense(filters)(layer)
        layer = BatchNormalization(momentum=0.0)(layer)
        return Activation("relu")(layer)

    def _conv1d_res_block(self, filters, kernel_size, block_input, first_block=False, time_related_enable=False):
        if first_block:
            block_input = self._conv1d_bn_relu(filters, kernel_size, block_input, time_related_enable)

        layer = self._conv1d_bn_relu(filters, kernel_size, block_input, time_related_enable)
        layer = self._conv1d_bn_relu(filters, kernel_size, layer, time_related_enable)

        if not time_related_enable:
            return Add()([layer, block_input])
        else:
            return TimeDistributed(Add())([layer, block_input])

    def _tnet(self, num_features, block_input, xyz_trans_only=False):  # based on PointNet structure
        if not xyz_trans_only:
            bias = keras.initializers.Constant(np.eye(num_features).flatten())  # Initalise bias as the indentity matrix
            reg = OrthogonalRegularizer(num_features)

            layer = self._conv1d_bn_relu(32, 1, block_input)
            layer = self._conv1d_bn_relu(64, 1, layer)
            layer = self._conv1d_bn_relu(512, 1, layer)
            layer = GlobalMaxPooling1D()(layer)
            layer = self._dense_bn_relu(256, layer)
            layer = self._dense_bn_relu(128, layer)
            layer = Dense(num_features * num_features,
                          kernel_initializer="zeros",
                          bias_initializer=bias,
                          activity_regularizer=reg,
                          )(layer)
            # calculate and apply affine transformation to input features
            feat_T = Reshape((num_features, num_features))(layer)
            return Dot(axes=(2, 1))([block_input, feat_T])
        else:
            num_features = 3
            # split xyz from v and snr
            split_layer = Lambda(lambda x: [x[:, :, :3], x[:, :, 3:]])
            split_output = split_layer(block_input)
            points_xyz = split_output[0]
            points_vsnr = split_output[1]

            bias = keras.initializers.Constant(np.eye(num_features).flatten())  # Initalise bias as the indentity matrix
            reg = OrthogonalRegularizer(num_features)

            layer = self._conv1d_bn_relu(32, 1, points_xyz)
            layer = self._conv1d_bn_relu(64, 1, layer)
            layer = self._conv1d_bn_relu(512, 1, layer)
            layer = GlobalMaxPooling1D()(layer)
            layer = self._dense_bn_relu(256, layer)
            layer = self._dense_bn_relu(128, layer)
            layer = Dense(num_features * num_features,
                          kernel_initializer="zeros",
                          bias_initializer=bias,
                          activity_regularizer=reg,
                          )(layer)
            # calculate and apply affine transformation to input features
            feat_T = Reshape((num_features, num_features))(layer)
            points_xyz = Dot(axes=(2, 1))([points_xyz, feat_T])
            return Concatenate(axis=-1)([points_xyz, points_vsnr])

    def _tnet_time(self, num_features, block_input, xyz_trans_only=False):
        if not xyz_trans_only:
            bias = keras.initializers.Constant(np.eye(num_features).flatten())  # Initalise bias as the indentity matrix
            reg = OrthogonalRegularizer(num_features)

            layer = self._conv1d_bn_relu(32, 1, block_input)
            layer = self._conv1d_bn_relu(64, 1, layer)
            layer = self._conv1d_bn_relu(256, 1, layer)
            layer = GlobalMaxPooling1D()(layer)
            layer = self._dense_bn_relu(128, layer)
            layer = self._dense_bn_relu(64, layer)
            layer = Dense(num_features * num_features,
                          kernel_initializer="zeros",
                          bias_initializer=bias,
                          activity_regularizer=reg,
                          )(layer)
            # calculate and apply affine transformation to input features
            feat_T = Reshape((num_features, num_features))(layer)
            return Dot(axes=(2, 1))([block_input, feat_T])
        else:
            num_features = 3
            # split xyz from v and snr
            layer = Reshape(target_shape=(-1, 5))(block_input)
            split_layer = Lambda(lambda x: [x[:, :, :3], x[:, :, 3:]])
            split_output = split_layer(layer)
            points_xyz = split_output[0]
            points_vsnr = split_output[1]

            bias = keras.initializers.Constant(np.eye(num_features).flatten())  # Initalise bias as the indentity matrix
            reg = OrthogonalRegularizer(num_features)

            layer = self._conv1d_bn_relu(32, 1, points_xyz)
            layer = self._conv1d_bn_relu(64, 1, layer)
            layer = self._conv1d_bn_relu(256, 1, layer)
            layer = GlobalMaxPooling1D()(layer)
            layer = self._dense_bn_relu(128, layer)
            layer = self._dense_bn_relu(64, layer)
            layer = Dense(num_features * num_features,
                          kernel_initializer="zeros",
                          bias_initializer=bias,
                          activity_regularizer=reg,
                          )(layer)
            # calculate and apply affine transformation to input features
            feat_T = Reshape((num_features, num_features))(layer)
            points_xyz = Dot(axes=(2, 1))([points_xyz, feat_T])
            layer = Concatenate(axis=-1)([points_xyz, points_vsnr])
            return Reshape(target_shape=(-1, 200, 5))(layer)

    """
    CNN main structures defined
    """

    def CNN1D_Dense(self, input_shape, output_shape):
        model_input = Input(shape=input_shape)

        layer = self._conv1d_bn_relu(128, 1, model_input)
        layer = MaxPooling1D(pool_size=2)(layer)
        layer = self._conv1d_bn_relu(128, 1, layer)
        layer = MaxPooling1D(pool_size=2)(layer)
        layer = self._conv1d_bn_relu(256, 1, layer)
        layer = MaxPooling1D(pool_size=2)(layer)
        layer = self._conv1d_bn_relu(256, 1, layer)
        layer = MaxPooling1D(pool_size=2)(layer)
        layer = self._conv1d_bn_relu(512, 1, layer)
        layer = MaxPooling1D(pool_size=2)(layer)
        layer = self._conv1d_bn_relu(512, 1, layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Flatten()(layer)
        layer = self._dense_bn_relu(512, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(256, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(64, layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(output_shape, activation='softmax')(layer)
        return Model(model_input, layer)

    def CNN1D_Dense_Tnet_GMPooling(self, input_shape, output_shape):
        model_input = Input(shape=input_shape)

        # layer = self._tnet(5, model_input, xyz_trans_only=False)
        layer = self._tnet(3, model_input, xyz_trans_only=True)  # xyz_trans_only will set num_features=3 and generate a 3*3 trans matrix for xyz dimensions only
        # layer = self._conv1d_bn_relu(64, 1, model_input)
        layer = self._conv1d_res_block(128, 1, layer, first_block=True)
        layer = self._conv1d_res_block(256, 1, layer, first_block=True)
        layer = self._conv1d_res_block(512, 1, layer, first_block=True)
        layer = GlobalMaxPooling1D()(layer)  # symmetric function for point cloud

        layer = self._dense_bn_relu(512, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(256, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(64, layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(output_shape, activation='softmax')(layer)
        return Model(model_input, layer)

    def CNN1D_Dense_LSTM(self, input_shape, output_shape):
        model_input = Input(shape=input_shape)

        layer = self._conv1d_bn_relu(128, 1, model_input, time_related_enable=True)
        layer = TimeDistributed(MaxPooling1D(pool_size=2))(layer)
        layer = self._conv1d_bn_relu(128, 1, layer, time_related_enable=True)
        layer = TimeDistributed(MaxPooling1D(pool_size=2))(layer)
        layer = self._conv1d_bn_relu(256, 1, layer, time_related_enable=True)
        layer = TimeDistributed(MaxPooling1D(pool_size=2))(layer)
        layer = self._conv1d_bn_relu(256, 1, layer, time_related_enable=True)
        layer = TimeDistributed(MaxPooling1D(pool_size=2))(layer)
        layer = self._conv1d_bn_relu(512, 1, layer, time_related_enable=True)
        layer = TimeDistributed(MaxPooling1D(pool_size=2))(layer)
        layer = self._conv1d_bn_relu(512, 1, layer, time_related_enable=True)
        layer = TimeDistributed(MaxPooling1D(pool_size=2))(layer)

        layer = TimeDistributed(Flatten())(layer)
        # layer = Bidirectional(LSTM(64, return_sequences=True))(layer)
        layer = Bidirectional(LSTM(32))(layer)
        # layer = LSTM(64, return_sequences=True)(layer)
        # layer = LSTM(128)(layer)

        layer = self._dense_bn_relu(256, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(128, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(64, layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(output_shape, activation='softmax')(layer)
        return Model(model_input, layer)

    def CNN1D_Dense_Tnet_GMPooling_LSTM(self, input_shape, output_shape):
        model_input = Input(shape=input_shape)

        layer = self._tnet_time(3, model_input, xyz_trans_only=True)  # xyz_trans_only will set num_features=3 and generate a 3*3 trans matrix for xyz dimensions only
        # layer = self._conv1d_bn_relu(64, 1, model_input, time_related_enable=True)

        layer = self._conv1d_res_block(128, 1, layer, first_block=True, time_related_enable=True)
        layer = self._conv1d_res_block(256, 1, layer, first_block=True, time_related_enable=True)
        layer = self._conv1d_res_block(512, 1, layer, first_block=True, time_related_enable=True)
        layer = TimeDistributed(GlobalMaxPooling1D())(layer)  # symmetric function for point cloud

        # layer = Bidirectional(LSTM(64, return_sequences=True))(layer)
        layer = Bidirectional(LSTM(32))(layer)
        # layer = LSTM(64, return_sequences=True)(layer)
        # layer = LSTM(128)(layer)

        layer = self._dense_bn_relu(256, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(128, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(64, layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(output_shape, activation='softmax')(layer)
        return Model(model_input, layer)

    def CNN1D_test(self, input_shape, output_shape):
        model_input = Input(shape=input_shape)

        # layer = self._tnet(5, model_input)
        layer = self._conv1d_bn_relu(64, 1, model_input)
        layer = self._conv1d_bn_relu(64, 1, layer)
        layer = self._conv1d_bn_relu(64, 1, layer)
        layer = self._conv1d_bn_relu(128, 1, layer)
        layer = self._conv1d_bn_relu(1024, 1, layer)
        layer = GlobalMaxPooling1D()(layer)  # symmetric function for point cloud

        layer = self._dense_bn_relu(512, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(256, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(64, layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(output_shape, activation='softmax')(layer)
        return Model(model_input, layer)

    def CNN2D_Dense(self, input_shape, output_shape):  # batch_size 256
        model_input = Input(shape=input_shape)

        layer = self._conv2d_bn_relu(512, (1, 1), model_input)
        layer = MaxPooling2D(pool_size=(2, 1))(layer)

        layer = self._conv2d_bn_relu(256, (1, 1), layer)
        layer = MaxPooling2D(pool_size=(2, 1))(layer)

        layer = self._conv2d_bn_relu(256, (1, 1), layer)
        layer = MaxPooling2D(pool_size=(2, 1))(layer)

        layer = self._conv2d_bn_relu(128, (1, 1), layer)
        layer = MaxPooling2D(pool_size=(2, 1))(layer)

        layer = self._conv2d_bn_relu(64, (1, 1), layer)
        layer = MaxPooling2D(pool_size=(2, 1))(layer)

        layer = Flatten()(layer)
        layer = self._dense_bn_relu(256, layer)
        layer = Dropout(0.3)(layer)
        layer = self._dense_bn_relu(64, layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(output_shape, activation='softmax')(layer)
        return Model(model_input, layer)


if __name__ == '__main__':
    # model = model_structure_5D((300, 5, 1), 10)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    model = CNN().CNN1D_test((300, 5), 10)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
