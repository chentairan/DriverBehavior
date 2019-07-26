from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.regularizers import l2

def AlexNet(input_layer, input_shape = (224, 224, 3), output_dim = 10 * 4096, l2_reg=0.0, weights=None):
    """
    AlexNet model
    :param input_layer: input layer
    :param input_shape: input shape
    :param num_classes: the number of classes
    :param l2_reg:
    :param weights:
    :return: model
    """

    # Layer 1
    # In order to get the same size of the paper mentioned, add padding layer first
    x = ZeroPadding2D(padding=(2, 2))(input_layer)
    x = conv_block(x, filters=96, kernel_size=(11, 11),
                   strides=(4, 4), padding="valid", l2_reg=l2_reg, name='Conv_1_96_11x11_4')
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_1_3x3_2")(x)

    # Layer 2
    x = conv_block(x, filters=256, kernel_size=(5, 5),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_2_256_5x5_1")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_2_3x3_2")(x)

    # Layer 3
    x = conv_block(x, filters=384, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_3_384_3x3_1")

    # Layer 4
    x = conv_block(x, filters=384, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_4_384_3x3_1")

    # Layer 5
    x = conv_block(x, filters=256, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_5_256_3x3_1")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_3_3x3_2")(x)

    # Layer 6
    x = Flatten()(x)
    x = Dense(units = output_dim)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', l2_reg=0.0, name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_regularizer=l2(l2_reg),
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x