import json
import pickle
from collections import deque, Counter

from tensorflow.keras.models import load_model
import tensorflow as tf

from NN import OrthogonalRegularizer
from config import *
from real_time_tracking.library.utils import *


def model_prediction(data_np, label_np, corresp_labels, model, print_details=False, blur_pred_len=False):
    acc_list = []
    acc_weight_list = []
    # predict for each person
    for person_label in list(corresp_labels.values()):
        person_data = data_np[label_np == person_label]
        person_labels = np.ones([len(person_data)]) * person_label

        # evaluate the model
        possibility = model.predict(person_data)
        pred_labels = np.argmax(possibility, axis=1)

        # blur predict
        if blur_pred_len:
            pred_labels_blur = np.empty(pred_labels.shape, dtype=np.int8)
            pred_labels_blur_deque = deque([], blur_pred_len)
            for idx, label in enumerate(pred_labels):
                pred_labels_blur_deque.append(label)
                if pred_labels_blur_deque.__len__() == pred_labels_blur_deque.maxlen:
                    label, _ = Counter(pred_labels_blur_deque).most_common(1)[0]
                    pred_labels_blur[idx] = label
                else:
                    pred_labels_blur[idx] = label
            pred_labels = pred_labels_blur

        # print each data piece result in file
        if print_details:
            for idx, pred_label in enumerate(pred_labels):
                print(f'[{pred_label}]: ', end='')
                print('[', end='')
                for poss in possibility[idx]:
                    print(f'{poss:.02f}, ', end='')
                print(']')
        # print the accuracy for this person
        accuracy = round(np.sum(pred_labels == person_labels) / len(person_labels), 3)
        name = str([key for key, value in corresp_labels.items() if value == person_label])
        print(f'People {name:<10} test accuracy: {accuracy:.03f}')
        acc_list.append(accuracy)
        acc_weight_list.append(round(len(person_data)/len(data_np), 3))
    print(f'{"Acc:":<10} {acc_list}')
    print(f'{"Weights:":<10} {acc_weight_list}')
    print(f'Overall accuracy: {np.sum(np.array(acc_list) * np.array(acc_weight_list)):.03f}')


if __name__ == '__main__':
    """evaluate test dataset"""
    # load corresponding labels
    with open(os.path.join(os.path.dirname(BESTMODEL_SAVE_DIR), '200_filtered_timelen_30_tnet/corresp_labels_12.json'), 'r') as json_file:
        corresp_labels = json.load(json_file)
    # load data and label
    dataset_file = 'test_dataset_12_200_filtered_timelen_30_step1'
    with open(f'{os.path.join(DATASET_DIR, dataset_file)}', 'rb') as file:
        test_data_np, test_labels_np = pickle.load(file)

    # test_data_np = test_data_np[:, :, :, np.newaxis]  # for 2D only
    # test_data_np = test_data_np[:, :, :, :, np.newaxis]  # for 2D time only

    # load model
    model = load_model(os.path.join(BESTMODEL_SAVE_DIR, '200_filtered_timelen_30_tnet/30.h5'), custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # when label_categorical is False
                  optimizer='adam',
                  metrics=['accuracy'])

    # # load checkpoint
    # model.load_weights(os.path.join(BESTMODEL_SAVE_DIR, '200_filtered_timelen_20/ep001_valloss1.12_valacc0.738.h5'))

    # predict each data file
    model_prediction(test_data_np, test_labels_np, corresp_labels, model, print_details=False, blur_pred_len=60)
