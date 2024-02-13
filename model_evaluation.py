import pickle

from tensorflow.keras.models import load_model

from NN import OrthogonalRegularizer
from config import *
from real_time_tracking.library.utils import *

if __name__ == '__main__':
    """evaluate test dataset"""
    # load data and label
    dataset_file = '200_filtered_timelen_30_tnet/test_dataset_200_filtered_timelen_10'
    with open(f'{os.path.join(BESTMODEL_SAVE_DIR, dataset_file)}', 'rb') as file:
        test_data_np, test_labels_np = pickle.load(file)

    # test_data_np = test_data_np[:, :, :, np.newaxis]  # for 2D only
    # test_data_np = test_data_np[:, :, :, :, np.newaxis]  # for 2D time only

    # load the model
    model = load_model(os.path.join(BESTMODEL_SAVE_DIR, '200_filtered_timelen_30_tnet/10.h5'), custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})
    # evaluate the model
    model.evaluate(test_data_np, test_labels_np, verbose=1)

    # # load the model
    # model = load_model(os.path.join(MODEL_SAVE_DIR, 'model_filtered.h5'), custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})
    # # evaluate the model
    # model.evaluate(test_data_np, test_labels_np, verbose=1)
