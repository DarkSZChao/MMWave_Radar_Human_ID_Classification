import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CP_DIR = os.path.join(ROOT_DIR, './model/checkpoint/')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, './model/')
BESTMODEL_SAVE_DIR = os.path.join(ROOT_DIR, './best_model/')

DATASET_DIR = os.path.join(ROOT_DIR, './data/')
DATASET_TRAIN_VAL_DIR = os.path.join(ROOT_DIR, './data/train_val/')
DATASET_TEST_DIR = os.path.join(ROOT_DIR, './data/test/')
