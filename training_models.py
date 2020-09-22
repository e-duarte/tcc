from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

class Trainner:
    def __init__(self,
        epochs=10,
        batch_size=None,
        data_augmentation=None,
        callbacks=[],
        dir_path='./',
        state=None):

        self.epochs=epochs

        self.batch_size=batch_size
        self.data_augmantion=data_augmentation
        self.callbacks=callbacks

    def train_model(self, x, y, model, validation_data=None, init_epoch=0):
        steps_per_epoch = int(len(x)/self.batch_size) if self.batch_size else None
    
        history = None
        if self.data_augmantion:
            history = model.fit(self.data_augmantion.flow(x, y, batch_size=self.batch_size),
                                epochs=self.epochs,
                                callbacks=self.callbacks,
                                validation_data=validation_data,
                                initial_epoch=init_epoch,
                            )
        else:
            history = model.fit(x,
                                y,
                                epochs=self.epochs,
                                callbacks=self.callbacks,
                                batch_size=self.batch_size,
                                validation_data=validation_data,
                                steps_per_epoch=steps_per_epoch,
                                initial_epoch=init_epoch,
                            )
        return history

