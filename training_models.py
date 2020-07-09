from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

class Trainner():
    def __init__(self,
        epochs=10,
        batch_size=None,
        data_augmentation=None,
        callbacks=[]):

        self.epochs=epochs

        self.batch_size=batch_size
        self.data_augmantion=data_augmentation
        self.callbacks=callbacks

        # checkpoint = ModelCheckpoint(checkpoint_name + '.hdf5', monitor='loss', verbose=1,
                                        # save_best_only=True, mode='auto', save_freq=save_freq)
        # self.callbacks.append(checkpoint)

    def train_model(self, x, y, model):
        steps_per_epoch = int(len(x)/self.batch_size) if self.batch_size else None
    
        history = None
        if self.data_augmantion:
            history = model.fit(self.data_augmantion.flow(x, y, batch_size=self.batch_size),
                        epochs=self.epochs,
                        callbacks=self.callbacks,
                        # use_multiprocessing=True,
                        steps_per_epoch=steps_per_epoch)
        else:
            history = model.fit(x,
                        y,
                        epochs=self.epochs,
                        # validation_split=0.1,
                        # callbacks=self.callbacks,
                        # use_multiprocessing=True,
                        batch_size=self.batch_size,
                        steps_per_epoch=steps_per_epoch)
        return history

