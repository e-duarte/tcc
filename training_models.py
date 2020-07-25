from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os


class CheckpointWeights(keras.callbacks.Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(filepath):
            print('making diretory...')
            os.mkdir(filepath)

    def on_epoch_end(self, epoch, logs=None):
        path = self.filepath + '{}_epochs_{}.h5'.format(self.model.name, epoch+1)

        if ((epoch+1)  % 100) == 0:
            print('\nsaving...{} - epochs:{}'.format(self.model.name, epoch+1))
            print(path, '\n')
            self.model.save_weights(path)

class Trainner():
    def __init__(self,
        epochs=10,
        batch_size=None,
        data_augmentation=None,
        callbacks=[],
        dir_path='./'):

        self.epochs=epochs

        self.batch_size=batch_size
        self.data_augmantion=data_augmentation
        self.callbacks=callbacks

        # checkpoint = ModelCheckpoint(checkpoint_name + '.hdf5', monitor='loss', verbose=1,
                                        # save_best_only=True, mode='auto', save_freq=save_freq)
        # self.callbacks.append(checkpoint)

        self.callbacks.append(CheckpointWeights(dir_path))

    def train_model(self, x, y, model, validation_data=None):
        steps_per_epoch = int(len(x)/self.batch_size) if self.batch_size else None
    
        history = None
        if self.data_augmantion:
            history = model.fit(self.data_augmantion.flow(x, y, batch_size=self.batch_size),
                        epochs=self.epochs,
                        callbacks=self.callbacks,
                        # use_multiprocessing=True,
                        validation_data=validation_data,
                        # steps_per_epoch=steps_per_epoch
                        )
        else:
            history = model.fit(x,
                        y,
                        epochs=self.epochs,
                        # validation_split=0.1,
                        callbacks=self.callbacks,
                        # use_multiprocessing=True,
                        batch_size=self.batch_size,
                        validation_data=validation_data,
                        steps_per_epoch=steps_per_epoch
                        )
        return history

