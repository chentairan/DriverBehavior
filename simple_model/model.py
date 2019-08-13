from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

def get_model():

    seq = Sequential(name='layer')
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                    input_shape=(None, 75, 100, 3),
                    padding='same', return_sequences=True, name='layer1'))
    seq.add(BatchNormalization(name='layer2'))

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                    padding='same', return_sequences=True, name='layer3'))
    seq.add(BatchNormalization(name='layer4'))

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                    padding='same', return_sequences=True, name='layer5'))
    seq.add(BatchNormalization(name='layer6'))

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                    padding='same', return_sequences=False, name='layer7'))
    seq.add(BatchNormalization(name='layer8'))

    seq.add(Flatten(name='layer9'))

    seq.add(Dense(2, activation='sigmoid', name='layer10'))

    seq.compile(loss='binary_crossentropy', optimizer='adadelta')

    return seq