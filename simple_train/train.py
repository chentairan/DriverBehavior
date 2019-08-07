"""
#This script demonstrates the use of a convolutional LSTM network.

This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.models import Sequential
from model import get_model
from data_process import DataGenerator
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import pylab as plt
import glob

num_classes = 2

model = get_model()

params = {'dim': (120,600,800),
          'batch_size': 6,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

# Datasets
risky_num = 3
safe_num = 3

partition = {}
partition['train'] = [i for i in range(risky_num + safe_num)]

labels = [0 for i in range(risky_num)] + [1 for i in range(safe_num)]

training_generator = DataGenerator(partition['train'], labels, **params)

# Train the network
# videos, labels = get_data()

model.fit_generator(epochs=30,
    generator=training_generator)

model.save_weights('my_model_weights.h5')