import glob
import json
import pickle

import numpy as np

from config import *
from real_time_tracking.library.utils import *


def gather_data(rawdata_dir, corresp_labels, padding_0_to_size=200, filter_enable=True, window_sliding_enable=False):
    # default params
    RADAR_CFG_LIST = [
        {'name'        : 'IWR1843_Ori',
         'ES_threshold': {'range': (150, None), 'speed_none_0_exception': True},  # if speed_none_0_exception is True, then the data with low ES but with speed will be reserved
         },
        {'name'        : 'IWR1843_Side',
         'ES_threshold': {'range': (150, None), 'speed_none_0_exception': True},  # if speed_none_0_exception is True, then the data with low ES but with speed will be reserved
         },
        {'name'        : 'IWR1843_Top',
         'ES_threshold': {'range': (120, None), 'speed_none_0_exception': True},  # if speed_none_0_exception is True, then the data with low ES but with speed will be reserved
         },
    ]
    window_sliding_len = 30
    window_sliding_step = 5

    people_np = np.empty([0, padding_0_to_size, 5], dtype=np.float16) if not window_sliding_enable else np.empty([0, window_sliding_len, padding_0_to_size, 5], dtype=np.float16)
    labels_list = []

    # load all people
    data_path_list = glob.glob(os.path.join(rawdata_dir, '*'))
    for idx, data_path in enumerate(data_path_list):
        # only take the data whose name appeared in json file
        name = os.path.basename(data_path).split('_')[0]
        if name in list(corresp_labels.keys()):
            # get the label for this file
            label = corresp_labels[name]

            with open(data_path, 'rb') as _file:
                data_person = pickle.load(_file)
                print(f'Working: {data_path}\t[{idx + 1}/{len(data_path_list)}]')
                # load all frames for each person
                data_person_np = np.empty([0, padding_0_to_size, 5], dtype=np.float16)
                for frame in data_person:
                    # get subframe from all radars
                    frames_np = np.empty([0, 5], dtype=np.float16)
                    for RADAR_CFG in RADAR_CFG_LIST:
                        frame_1radar = frame[RADAR_CFG['name']]
                        if filter_enable:
                            # snr filter
                            frame_1radar, _ = ES_speed_filter(frame_1radar, RADAR_CFG['ES_threshold'])
                            # some data are with ridiculous high SNR level, pls filter them
                            # if len(frame_1radar) != 0 and frame_1radar[:, 4].max() > 1000:
                            #     print(RADAR_CFG['name'], frame_1radar[:, 4].max())
                            frame_1radar, _ = np_filter(frame_1radar, axis=4, range_lim=(None, 1000))
                            # boundary filter
                            frame_1radar, _ = np_filter(frame_1radar, axis=0, range_lim=(-1.7, 1.6))
                            frame_1radar, _ = np_filter(frame_1radar, axis=1, range_lim=(0, 3.6))
                            frame_1radar, _ = np_filter(frame_1radar, axis=2, range_lim=(0.1, 2))

                        # data shift and standardization
                        # shift x dimension
                        frame_1radar[:, 0] = frame_1radar[:, 0] + 1.7
                        # abs the speed dimension
                        frame_1radar[:, 3] = np.abs(frame_1radar[:, 3])
                        # standardize the dimensions
                        frame_1radar[:, 3] = frame_1radar[:, 3] * 2
                        frame_1radar[:, 4] = frame_1radar[:, 4] / 200

                        # gather frames
                        frames_np = np.concatenate([frames_np, frame_1radar])
                    # padding 0
                    diff = padding_0_to_size - len(frames_np)
                    frames_np = np.concatenate([frames_np, np.zeros([diff, frames_np.shape[1]], dtype=np.float16)])

                    # form the data
                    data_person_np = np.concatenate([data_person_np, frames_np[np.newaxis, :, :]])
                    # form the label
                    if not window_sliding_enable:
                        labels_list.append(label)

            if window_sliding_enable:
                data_person_np = np_window_sliding(data_person_np, window_sliding_len, window_sliding_step)
                for i in range(len(data_person_np)):
                    labels_list.append(label)

            people_np = np.concatenate([people_np, data_person_np])
            pass

    return people_np.astype(np.float16), np.array(labels_list).astype(np.int8)


if __name__ == '__main__':
    # load corresponding labels
    with open(os.path.join(os.path.dirname(DATASET_DIR), 'corresp_labels_12.json'), 'r') as json_file:
        name_labels = json.load(json_file)

    # data_np, labels_np = gather_data(DATASET_TRAIN_VAL_DIR, name_labels, padding_0_to_size=300, filter_enable=False)
    # dataset_file = 'train_val_dataset_300'

    # data_np, labels_np = gather_data(DATASET_TEST_DIR, name_labels, padding_0_to_size=300, filter_enable=False)
    # dataset_file = 'test_dataset_300'

    # data_np, labels_np = gather_data(DATASET_TRAIN_VAL_DIR, name_labels, padding_0_to_size=300, filter_enable=False, window_sliding_enable=True)
    # dataset_file = 'train_val_dataset_300_timelen_20'

    # data_np, labels_np = gather_data(DATASET_TEST_DIR, name_labels, padding_0_to_size=300, filter_enable=False, window_sliding_enable=True)
    # dataset_file = 'test_dataset_300_timelen_20'

    # data_np, labels_np = gather_data(DATASET_TRAIN_VAL_DIR, name_labels, padding_0_to_size=200, filter_enable=True)
    # dataset_file = 'train_val_dataset_200_filtered'

    # data_np, labels_np = gather_data(DATASET_TEST_DIR, name_labels, padding_0_to_size=200, filter_enable=True)
    # dataset_file = 'test_dataset_200_filtered'

    # data_np, labels_np = gather_data(DATASET_TRAIN_VAL_DIR, name_labels, padding_0_to_size=200, filter_enable=True, window_sliding_enable=True)
    # dataset_file = 'train_val_dataset_200_filtered_timelen_30'

    data_np, labels_np = gather_data(DATASET_TEST_DIR, name_labels, padding_0_to_size=200, filter_enable=True, window_sliding_enable=True)
    dataset_file = 'test_dataset_200_filtered_timelen_30'

    with open(os.path.join(DATASET_DIR, dataset_file), 'wb') as file:
        pickle.dump((data_np, labels_np), file)
    print(f'{os.path.join(DATASET_DIR, dataset_file)} is saved')

    # with open(os.path.join(os.path.dirname(DATASET_DIR), 'train_val_dataset_200_filtered'), 'rb') as file:
    #     data, label = pickle.load(file)
    #     pass
