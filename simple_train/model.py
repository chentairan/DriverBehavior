from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

def get_model():

    seq = Sequential()
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                    input_shape=(None, 395, 800, 3),
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

    seq.add(Dense(2, activation='sigmoid'))

    seq.compile(loss='binary_crossentropy', optimizer='adadelta')

    return seq