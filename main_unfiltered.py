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


if __name__ == '__main__':
    # load data from stored dataset
    dataset_file = 'train_val_dataset'
    with open(f'{os.path.join(DATASET_DIR, dataset_file)}', 'rb') as file:
        data_np, labels_np = pickle.load(file)

    # generate the dataset for train and val
    train_data_np, train_labels_np, val_data_np, val_labels_np, category_No = generate_dataset(data_np, labels_np, split_ratio=(0.8, 0.2), label_categorical=False, test_data_enable=False)

    # clear the last checkpoints
    folder_clean_recreate(CP_DIR)

    # create the model
    NN = CNN()
    model = NN.CNN1D_res(train_data_np.shape[1:], category_No)
    # train_data_np = train_data_np[:, :, :, np.newaxis]  # for 2D only
    # val_data_np = val_data_np[:, :, :, np.newaxis]  # for 2D only
    # model = NN.CNN2D(train_data_np.shape[1:], category_No)

    # model.compile(loss='categorical_crossentropy',  # when label_categorical is True
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # when label_categorical is False
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_data_np, train_labels_np,
                        validation_data=(val_data_np, val_labels_np),
                        shuffle=True,
                        batch_size=512,
                        callbacks=[NN.lr_callback, NN.cp_callback],
                        epochs=40,
                        verbose=1)

    # save model
    model.save(os.path.join(MODEL_SAVE_DIR, 'model.h5'))

    # print results
    print(f"Epoch {np.argmax(history.history['val_accuracy']) + 1} with maximum val_accuracy: {history.history['val_accuracy'][np.argmax(history.history['val_accuracy'])] * 100:.2f}%")

    # Loads the best weights
    epoch_remove = int(len(history.epoch) / 4)  # avoid to pick the first several checkpoints
    epoch_val_acc_list = history.history['val_accuracy'][epoch_remove:]
    h5file = glob.glob(os.path.join(CP_DIR, f'ep{np.argmax(epoch_val_acc_list) + 1 + epoch_remove:03d}_*.h5'))[0]
    print(f'Loading weight checkpoint: {os.path.basename(h5file)}')
    model.load_weights(h5file)

    # save model
    model.save(os.path.join(MODEL_SAVE_DIR, 'modelbest.h5'))

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
