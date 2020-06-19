from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

class Trainner():
    def __init__(self,model,
        dataset,
        checkpoint_name='model_save',
        epochs=10,
        validation_data=None,
        batch_size=None,
        data_augmentation=None,
        callbacks=[],
        tensor_board=False):

        self.model = model
        self.x = dataset[0]
        self.y = dataset[1]
        self.epochs=epochs

        self.validation_data=validation_data
        self.batch_size=batch_size
        self.data_augmantion=data_augmentation
        self.callbacks=callbacks

        self.steps_per_epoch = int(len(self.x)/self.batch_size) if batch_size else None
        save_freq = self.steps_per_epoch if batch_size else len(self.x)

        if tensor_board:
            self.callbacks.append(TensorBoard(log_dir='./{}'.format(checkpoint_name), histogram_freq=1))

        checkpoint = ModelCheckpoint(checkpoint_name + '.hdf5', monitor='loss', verbose=1,
                                        save_best_only=True, mode='auto', save_freq=save_freq)
        self.callbacks.append(checkpoint)

    def train_model(self):
        history = None
        if self.data_augmantion:
            history = self.model.fit(self.data_augmantion.flow(self.x, self.y, batch_size=self.batch_size),
                        epochs=self.epochs,
                        callbacks=self.callbacks,
                        # use_multiprocessing=True,
                        steps_per_epoch=self.steps_per_epoch,
                        validation_data=self.validation_data,)
        else:
            history = self.model.fit(self.x,
                        self.y,
                        epochs=self.epochs,
                        # callbacks=self.callbacks,
                        # use_multiprocessing=True,
                        batch_size=self.batch_size,
                        steps_per_epoch=self.steps_per_epoch,
                        validation_data=self.validation_data)
        return history



