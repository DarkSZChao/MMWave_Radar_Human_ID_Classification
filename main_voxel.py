import glob
import pickle

from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

from NN import *
from config import *
from real_time_tracking.library.utils import *

# enable GPU for calculation
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 0 for GPU, -1 for CPU
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))


def generate_dataset(_data_np, _labels_np, split_ratio, label_categorical=False, test_data_enable=False):
    if not test_data_enable:
        # define dataset params
        _category_No = max(_labels_np) + 1
        _train_data_np = np.empty([0] + list(_data_np.shape[1:]), dtype=np.float16)
        _train_labels_np = np.empty([0] + list(_labels_np.shape[1:]), dtype=np.int8)
        _val_data_np = np.empty([0] + list(_data_np.shape[1:]), dtype=np.float16)
        _val_labels_np = np.empty([0] + list(_labels_np.shape[1:]), dtype=np.int8)
        for category in range(_category_No):
            # split sub-dataset from each category into train and val
            idx = np.squeeze(np.argwhere(_labels_np == category))
            train_idx, val_idx = dataset_split(idx, split_ratio=split_ratio, random=False)
            _train_data_np = np.concatenate([_train_data_np, _data_np[train_idx]])
            _train_labels_np = np.concatenate([_train_labels_np, _labels_np[train_idx]])
            _val_data_np = np.concatenate([_val_data_np, _data_np[val_idx]])
            _val_labels_np = np.concatenate([_val_labels_np, _labels_np[val_idx]])

        if label_categorical:
            return _train_data_np, to_categorical(_train_labels_np).astype(np.int8), \
                _val_data_np, to_categorical(_val_labels_np).astype(np.int8), \
                _category_No
        else:
            return _train_data_np, _train_labels_np[:, np.newaxis], \
                _val_data_np, _val_labels_np[:, np.newaxis], \
                _category_No
    else:
        # define dataset params
        _category_No = max(_labels_np) + 1
        _train_data_np = np.empty([0] + list(_data_np.shape[1:]), dtype=np.float16)
        _train_labels_np = np.empty([0] + list(_labels_np.shape[1:]), dtype=np.int8)
        _val_data_np = np.empty([0] + list(_data_np.shape[1:]), dtype=np.float16)
        _val_labels_np = np.empty([0] + list(_labels_np.shape[1:]), dtype=np.int8)
        _test_data_np = np.empty([0] + list(_data_np.shape[1:]), dtype=np.float16)
        _test_labels_np = np.empty([0] + list(_labels_np.shape[1:]), dtype=np.int8)
        for category in range(_category_No):
            # split sub-dataset from each category into train and val
            idx = np.squeeze(np.argwhere(_labels_np == category))
            train_idx, val_idx, test_idx = dataset_split(idx, split_ratio=split_ratio, random=False)
            _train_data_np = np.concatenate([_train_data_np, _data_np[train_idx]])
            _train_labels_np = np.concatenate([_train_labels_np, _labels_np[train_idx]])
            _val_data_np = np.concatenate([_val_data_np, _data_np[val_idx]])
            _val_labels_np = np.concatenate([_val_labels_np, _labels_np[val_idx]])
            _test_data_np = np.concatenate([_test_data_np, _data_np[test_idx]])
            _test_labels_np = np.concatenate([_test_labels_np, _labels_np[test_idx]])

        if label_categorical:
            return _train_data_np, to_categorical(_train_labels_np).astype(np.int8), \
                _val_data_np, to_categorical(_val_labels_np).astype(np.int8), \
                _test_data_np, to_categorical(_test_labels_np).astype(np.int8), \
                _category_No
        else:
            return _train_data_np, _train_labels_np[:, np.newaxis], \
                _val_data_np, _val_labels_np[:, np.newaxis], \
                _test_data_np, _test_labels_np[:, np.newaxis], \
                _category_No


def matrix_generator(size=None, resolution=None, coords=None, value=None):
    """
    only support 3D cube matrix
    :param size: (tuple/list) xyz in meters
    :param resolution: (float) the matrix resolution
    :param coords: (nparray) containing xyz numbers
    :param value: (nparray) containing the value of that position
    :return: _matrix: (ndarray) 3D cube matrix
    """
    # create an empty matrix of zeros based on the size and resolution
    _matrix = np.zeros([int(i / resolution) for i in size])
    # set values based on coords
    for c, v in zip(coords, value):
        x = int(c[0] / resolution)
        y = int(c[1] / resolution)
        z = int(c[2] / resolution)
        if _matrix[x, y, z] == 0:
            _matrix[x, y, z] = v
        else:
            _matrix[x, y, z] = (_matrix[x, y, z] + v) / 2  # get average if 2 or more points set into one mesh

    return _matrix


def dataset_length_fixer(method, _alldata_list):
    if method == '3D_matrix_cube':
        matrix_size = (2, 4, 3)
        matrix_resolution = 0.1
        matrix_mesh_No = [int(i / matrix_resolution) for i in matrix_size]
        _matrix_vel_np = np.empty([0] + matrix_mesh_No)
        _matrix_SNR_np = np.empty([0] + matrix_mesh_No)
        i = 0
        for frame in _alldata_list:
            i += 1
            print('Processing Frame No:', i)

            # split coords and info
            coords = frame[:, :3]
            velocity = frame[:, 3]
            SNR = frame[:, 4]

            # shift the coords, keep positive
            coords_mapped = ((coords + (1, 0, 1)) * (1, 1, 1)).astype(np.float16)

            # generate sparse matrix
            matrix_vel = matrix_generator(size=matrix_size, resolution=matrix_resolution, coords=coords_mapped, value=velocity)[np.newaxis, :]
            matrix_SNR = matrix_generator(size=matrix_size, resolution=matrix_resolution, coords=coords_mapped, value=SNR)[np.newaxis, :]
            # merge all matrix
            _matrix_vel_np = np.concatenate([_matrix_vel_np, matrix_vel], axis=0)
            _matrix_SNR_np = np.concatenate([_matrix_SNR_np, matrix_SNR], axis=0)

        return _matrix_vel_np[:, :, :, :, np.newaxis], _matrix_SNR_np[:, :, :, :, np.newaxis]

    elif method == '3D_matrix_cube_small':  # coords start with (0, 0, 0) by default
        matrix_size = (0.8, 0.8, 1.8)
        matrix_central_point = [i / 2 for i in matrix_size]
        matrix_resolution = 0.1
        matrix_mesh_No = [int(i / matrix_resolution) for i in matrix_size]
        _matrix_vel_np = np.empty([0] + matrix_mesh_No)
        _matrix_SNR_np = np.empty([0] + matrix_mesh_No)
        count = 0
        for frame in _alldata_list:
            count += 1
            print('Processing Frame No:', count)

            # split coords and info
            coords = frame[:, :3]
            velocity = frame[:, 3]
            SNR = frame[:, 4]

            # find current frame weight central point
            point_No = frame.shape[0]
            x = sum(coords[:, 0]) / point_No
            y = sum(coords[:, 1]) / point_No
            z = sum(coords[:, 2]) / point_No
            data_central_point = [x, y, z]
            # compare with matrix central point and map to it
            shift_diff = [m - d for m, d in zip(matrix_central_point, data_central_point)]
            coords_mapped = (coords + shift_diff).astype(np.float16)

            # remove points outside
            coords_mapped = coords_mapped[(coords_mapped[:, 0] >= 0) & (coords_mapped[:, 0] < matrix_size[0])]
            coords_mapped = coords_mapped[(coords_mapped[:, 1] >= 0) & (coords_mapped[:, 1] < matrix_size[1])]
            coords_mapped = coords_mapped[(coords_mapped[:, 2] >= 0) & (coords_mapped[:, 2] < matrix_size[2])]

            # generate sparse matrix
            matrix_vel = matrix_generator(size=matrix_size, resolution=matrix_resolution, coords=coords_mapped, value=velocity)[np.newaxis, :]
            matrix_SNR = matrix_generator(size=matrix_size, resolution=matrix_resolution, coords=coords_mapped, value=SNR)[np.newaxis, :]
            # merge all matrix
            _matrix_vel_np = np.concatenate([_matrix_vel_np, matrix_vel], axis=0)
            _matrix_SNR_np = np.concatenate([_matrix_SNR_np, matrix_SNR], axis=0)

        return _matrix_vel_np[:, :, :, :, np.newaxis], _matrix_SNR_np[:, :, :, :, np.newaxis]


if __name__ == '__main__':
    # load data from stored dataset
    dataset_file = 'train_val_dataset_300_timelen_20'
    with open(f'{os.path.join(DATASET_DIR, dataset_file)}', 'rb') as file:
        data_np, labels_np = pickle.load(file)

    # generate the dataset for train and val
    train_data_np, train_labels_np, val_data_np, val_labels_np, category_No = generate_dataset(data_np, labels_np, split_ratio=(0.8, 0.2), label_categorical=False, test_data_enable=False)

    # clear the last checkpoints
    folder_clean_recreate(CP_DIR)

    # create the model
    NN = CNN()
    model = NN.CNN1D_time(train_data_np.shape[1:], category_No)
    # train_data_np = train_data_np[:, :, :, np.newaxis]  # for 2D only
    # val_data_np = val_data_np[:, :, :, np.newaxis]  # for 2D only
    # model = NN.CNN2D(train_data_np.shape[1:], category_No)
    # train_data_np = train_data_np[:, :, :, :, np.newaxis]  # for 2D time only
    # val_data_np = val_data_np[:, :, :, :, np.newaxis]  # for 2D time only
    # model = NN.CNN1D_time2(train_data_np.shape[1:], category_No)

    # model.compile(loss='categorical_crossentropy',  # when label_categorical is True
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # when label_categorical is False
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    history = model.fit(train_data_np, train_labels_np,
                        validation_data=(val_data_np, val_labels_np),
                        shuffle=True,
                        batch_size=2,
                        callbacks=[NN.lr_callback, NN.cp_callback],
                        epochs=20,
                        verbose=1)

    # save model
    model.save(os.path.join(MODEL_SAVE_DIR, 'model_filtered.h5'))

    # print results
    print(f"Epoch {np.argmax(history.history['val_accuracy']) + 1} with maximum val_accuracy: {history.history['val_accuracy'][np.argmax(history.history['val_accuracy'])] * 100:.2f}%")

    # Loads the best weights
    epoch_remove = int(len(history.epoch) / 4)  # avoid to pick the first several checkpoints
    epoch_val_acc_list = history.history['val_accuracy'][epoch_remove:]
    h5file = glob.glob(os.path.join(CP_DIR, f'ep{np.argmax(epoch_val_acc_list) + 1 + epoch_remove:03d}_*.h5'))[0]
    print(f'Loading weight checkpoint: {os.path.basename(h5file)}')
    model.load_weights(h5file)

    # save model
    model.save(os.path.join(MODEL_SAVE_DIR, 'modelbest_filtered.h5'))

    # draw the figures
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
