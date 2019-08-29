import numpy as np
import keras
from PIL import Image

label_colormap = {
    "unlabeled":(0,0,0),
    "ego vehicle":(0,0,0),
    "rectification border":(0,0,0),
    "out of roi":(0,0,0),
    "static":(0,0,0),
    "dynamic":(111,74,0),
    "ground":(81,0,81),
    "road":(128,64,128),
    "sidewalk":(244, 35,232),
    "parking":(250,170,160),
    "rail track":(230,150,140),
    "building":( 70, 70, 70),
    "wall":(102,102,156),
    "fence":(190,153,153),
    "guard rail":(180,165,180),
    "bridge":(150,100,100),
    "tunnel":(150,120, 90),
    "pole":(153,153,153),
    "polegroup":(153,153,153),
    "traffic light":(250,170,30),
    "traffic sign":(220,220,0),
    "vegetation":(107,142,35),
    "terrain":(152,251,152),
    "sky":(70,130,180),
    "person":(220, 20,60),
    "rider":(255,0,0),
    "car":(0,0,142),
    "truck":(0,0,70),
    "bus":(0,60,100),
    "caravan":(0,0,90),
    "trailer":(0,0,110),
    "train":(0,80,100),
    "motorcycle":(0,0,230),
    "bicycle":(119,11,32),
    "license plate":(0,0,142)
}

id2label = [
    "unlabeled",
    "ego vehicle",
    "rectification border",
    "out of roi",
    "static",
    "dynamic",
    "ground",
    "road",
    "sidewalk",
    "parking",
    "rail track",
    "building",
    "wall",
    "fence",
    "guard rail",
    "bridge",
    "tunnel",
    "pole",
    "polegroup",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "train",
    "motorcycle",
    "bicycle",
    "license plate"
]

label2id = {label:id for id, label in enumerate(id2label)}
idcolormap = [label_colormap[label] for label in id2label]

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size = 32, dim = (128, 96), n_channels = 3, shuffle = True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
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
        Y = np.empty((self.batch_size, *self.dim, 1))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            temp = ID.replace('gtFine_trainvaltest', 'leftImg8bit_trainvaltest')
            temp = temp.replace('gtFine_color', 'leftImg8bit')
            temp = temp.replace('gtFine', 'leftImg8bit')
            img = np.array(Image.open(temp).resize((320, 240), resample = Image.CUBIC))
            label = np.array(Image.open(ID).convert("RGB").resize((320, 240), resample = Image.NEAREST))
            label = self.rgb2seglabel(label, idcolormap, True)

            X[i] = img
            # Store class
            Y[i] = label

        return X, Y


    def rgb2seglabel(self, img, colormap, channels_axis=False):
        """ Given an RGB image stored as a numpy array, and a colormap that
            maps from label id to an RGB color for that label, it returns a
            new numpy array with color chanel size of 1 where the pixel
            intensity values represent the class label id.
        Args:
            img:            (np array)
            colormap:       (list) list of pixel values for each class label id
            channels_axis:  (bool)(default=False) Should it return an array with a
                            third (color channels) axis of size 1?
        """
        height, width, _ = img.shape
        if channels_axis:
            label = np.zeros([height, width,1], dtype=np.uint8)
        else:
            label = np.zeros([height, width], dtype=np.uint8)
        for id in range(len(colormap)):
            label[np.all(img==np.array(idcolormap[id]), axis=2)] = id
        return label