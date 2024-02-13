import glob
import pickle

from NN import *

if __name__ == '__main__':
    # # load model
    # model = load_model(os.path.join('', '../best_model/200_filtered_timelen_20.h5'), custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})

    # load model
    dataset_file = 'test_dataset_200_filtered_timelen_10'
    with open(f'{os.path.join(DATASET_DIR, dataset_file)}', 'rb') as file:
        test_data_np, test_labels_np = pickle.load(file)

    NN = CNN()
    model = NN.CNN1D_time(test_data_np.shape[1:], 12)
    # model.compile(loss='categorical_crossentropy',  # when label_categorical is True
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # when label_categorical is False
                  optimizer='adam',
                  metrics=['accuracy'])

    # load checkpoint
    model.load_weights(glob.glob(os.path.join(BESTMODEL_SAVE_DIR, '200_filtered_timelen_30_tnet/ep003_trainacc_0.966_valloss0.97_valacc0.780.h5'))[-1])

    # save model
    model.save('szc.h5')
