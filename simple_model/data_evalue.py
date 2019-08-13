import numpy as np
import keras
from PIL import Image
import glob

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size = 32, dim = (120, 600, 800), n_channels = 3,
                 n_classes = 2, shuffle = True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        n_image = 50

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if ID > 150:    #safe one
                for j in range(120-n_image,120):
                    imageFile = '/content/drive/My Drive/new_data' + '/' + str(ID+1) + '/' + str(j) + '.jpg'
                    img = np.array(Image.open(imageFile))
                    X[i,j-(120-n_image)] = img
            else:
                file = '/content/drive/My Drive/new_data/' + str(ID+1)
                num_image = len(glob.glob(pathname = file + '/*'))
                for j in range(num_image-n_image, num_image):
                    imageFile = '/content/drive/My Drive/new_data' + '/' + str(ID+1) + '/' + str(j) + '.jpg'
                    img = np.array(Image.open(imageFile))
                    X[i,j-(num_image-n_image)] = img
            # Store class
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
