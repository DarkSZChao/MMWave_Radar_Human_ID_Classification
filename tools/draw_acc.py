import glob
import os
import pickle

from matplotlib import pyplot as plt

from config import MODEL_SAVE_DIR

file_list = glob.glob(os.path.join(MODEL_SAVE_DIR, 'system_comparsion/model_filtered_history_*'))

history_list = []
para_list = []

for file in file_list:
    with open(file, 'rb') as f:
        history_list.append(pickle.load(f))
        para_list.append(file.split('history_')[-1])


# draw the figures
plt.figure(figsize=(6.4, 4.8))
for history, para in zip(history_list, para_list):
    # plt.plot(history['accuracy'], label=f'accuracy_{para}')
    # plt.plot(history['val_accuracy'], label=f'val_accuracy_{para}')
    plt.plot(history['val_accuracy'], label=f'{para}')
plt.title('val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')


plt.figure(figsize=(6.4, 4.8))
for history, para in zip(history_list, para_list):
    # plt.plot(history['loss'], label=f'loss_{para}')
    # plt.plot(history['val_loss'], label=f'val_loss_{para}')
    plt.plot(history['val_loss'], label=f'{para}')
plt.title('val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 6])
plt.legend(loc='upper right')


plt.show()
