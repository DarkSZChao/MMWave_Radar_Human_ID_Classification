import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CP_DIR = os.path.join(ROOT_DIR, './model/checkpoint/')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, './model/')
BESTMODEL_SAVE_DIR = os.path.join(ROOT_DIR, './best_model/')

DATASET_DIR = os.path.join(ROOT_DIR, './data/')
RAW_DATA_DIR = os.path.join(ROOT_DIR, './data/raw_data/')
RAW_DATA_TRAIN_VAL_DIR = os.path.join(ROOT_DIR, './data/raw_data/train_val/')
RAW_DATA_TEST_DIR = os.path.join(ROOT_DIR, './data/raw_data/test/')
