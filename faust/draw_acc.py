import glob
import pickle

from matplotlib import pyplot as plt

file_list = glob.glob('./results/*cnn*')

history_list = []
para_list = []

for file in file_list:
    with open(file, 'rb') as f:
        history_list.append(pickle.load(f))
        para_list.append(f"~{str(round(int(file.split('_')[-1]) / 1000))}k")

para_list = para_list[::-1]


# draw the figures
plt.figure()
for history, para in zip(history_list, para_list):
    # plt.plot(history['accuracy'], label=f'accuracy_{para}')
    # plt.plot(history['val_accuracy'], label=f'val_accuracy_{para}')
    plt.plot(history['val_accuracy'], label=f'{para}')
plt.title('val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='upper right')


plt.figure()
for history, para in zip(history_list, para_list):
    # plt.plot(history['loss'], label=f'loss_{para}')
    # plt.plot(history['val_loss'], label=f'val_loss_{para}')
    plt.plot(history['val_loss'], label=f'{para}')
plt.title('val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')


plt.show()
