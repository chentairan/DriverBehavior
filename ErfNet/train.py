from ErfNet import ERFNet
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import pylab as plt
import glob
from get_data import get_data
from keras.preprocessing.image import ImageDataGenerator

model = ERFNet((240, 320, 3))._get_model(False)

filepath='/content/drive/My Drive/weights1.best.h5'

model.load_weights('/content/drive/My Drive/weights.best.h5')

checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True)

callbacks_list = [checkpointer]


#get data
train_file = '/content/drive/My Drive/data/gtFine/train'
val_file = '/content/drive/My Drive/data/gtFine/val'

partition = {}
partition['train'] = glob.glob(pathname = train_file + '/*/*gtFine_color.png')
partition['evaluate'] = glob.glob(pathname = val_file + '/*/*gtFine_color.png')

X_train, Y_train = get_data(partition['train'])
X_eval, Y_eval = get_data(partition['evaluate'])

train_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_gen.fit(X_train)

eval_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_gen.fit(X_eval)

print("training")
model.fit
model.fit_generator(generator = train_gen.flow(X_train, Y_train, batch_size=16),
                            steps_per_epoch=len(X_train) / 16,
                            epochs=80, 
                            use_multiprocessing=True,
                            callbacks=callbacks_list,
                            validation_data = eval_gen.flow(X_eval, Y_eval, batch_size=16),
                            validation_steps=len(X_eval) / 16)

# training_generator = DataGenerator(partition['train'], **params)
# evaluate_generator = DataGenerator(partition['evaluate'], **params)



# model.fit_generator(epochs=50,
#                     workers=16,
#                     use_multiprocessing=True,
#                     generator=training_generator,
#                     validation_data = evaluate_generator,
#                     callbacks=callbacks_list)

model.save_weights('/content/drive/My Drive/final_weights3.h5')