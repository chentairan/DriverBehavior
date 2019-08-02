from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from alexnet_base import AlexNet

#spatial

spatial_input = Input(shape=(224,224,3), name='Spatial')

x1 = AlexNet(spatial_input)


#temporal

temporal_input = Input(shape=(224,224,2), name='Temporal')

x2 = AlexNet(input_layer = spatial_input, input_shape = (224, 224, 2), output_dim = 9 * 4096)



x = concatenate([x1,x2])