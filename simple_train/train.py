"""
#This script demonstrates the use of a convolutional LSTM network.

This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.models import Sequential

from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import pylab as plt
import glob

num_classes = 2

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(50, 395, 800, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=False))
seq.add(BatchNormalization())

seq.add(Flatten())

seq.add(Dense(1, activation='sigmoid'))

seq.compile(loss='binary_crossentropy', optimizer='adadelta')

def get_data():
    risky_dir = '/Users/chentairan/Desktop/UCLA SMERC/Code/simple_train/dataset_risky'
    safe_dir = '/Users/chentairan/Desktop/UCLA SMERC/Code/simple_train/dataset_safe'

    risky_path_file=glob.glob(pathname = risky_dir + '/*')
    safe_path_file=glob.glob(pathname = safe_dir + '/*')

    
    labels = np.zeros((len(safe_path_file)+len(risky_path_file), 1), dtype=np.int)

    count = 0

    for i, file in enumerate(risky_path_file):
        num_image = len(glob.glob(pathname = file + '/*'))
        risky_videos = np.zeros((len(risky_path_file), 50, 395, 800, 3), dtype=np.float)
        images = np.zeros((0, 395, 800, 3), dtype=np.float)
        for j in range(num_image):
            imageFile = file + '/' + str(j) + '.jpg'
            img = np.array(Image.open(imageFile))
            images = np.vstack((images,[img]))

        images = pad_sequences([images], maxlen=50)

        risky_videos[i] = images
      
        labels[count][0] = 0
        count += 1

    for i, file in enumerate(safe_path_file):
        num_image = len(glob.glob(pathname = file + '/*'))
        safe_videos = np.zeros((len(safe_path_file), 50, 395, 800, 3), dtype=np.float)
        images = np.zeros((0, 395, 800, 3), dtype=np.float)
        for j in range(num_image):
            imageFile = file + '/' + str(j) + '.jpg'
            img = np.array(Image.open(imageFile))
            images = np.vstack((images,[img]))
        
        images = pad_sequences([images], maxlen=50)

        safe_videos[i] = images

        labels[count][0] = 1
        count += 1
    
    return np.vstack((risky_videos,safe_videos)), labels

# Train the network
videos, labels = get_data()

seq.fit(videos, labels, batch_size=10, epochs=1, validation_split=0.05)

seq.save_weights('my_model_weights.h5')