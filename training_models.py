from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os, random, string
from save import SaveExperiment
import pandas as pd

class CheckpointState(keras.callbacks.Callback):
    def __init__(self, filepath, state, epochs=None):
        self.filepath = filepath
        self.state = state
        self.epochs = epochs
        self.save = SaveExperiment(root_dir=filepath+'state/tmp/')
        if not os.path.exists(filepath):
            print('making diretory...')
            os.mkdir(filepath)

    def on_epoch_end(self, epoch, logs=None):
        name_weights = '{}_{}_{}_epochs_{}.h5'.format(self.state.dataset_name,
                                                            self.model.name,
                                                            self.state.exp,
                                                            epoch+1)
        path = self.filepath + 'weights/' + name_weights

        history_path = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(10))

        history = self.model.history.history
        print('\nsaving Temp History...\n')
        if epoch+1 != 1:
            if self.state.exp.split('_')[0] == 'k-fold':
                self.state.update_history_k(self.state.last_k() - 1, self.filepath + 'state/tmp/' + history_path + '.csv')
            else:
                self.state.update_history_h(self.filepath + 'state/tmp/' + history_path + '.csv')

            history['epochs'].append(epoch)
            self.save.save_history_csv(history, history_path)
            self.state.save_state()
        else:
            history['epochs'] = []
        
        if self.epochs == epoch + 1:
            dir_type = 'results/conv/{}/'.format(self.model.name.split('_')[0]) if self.state.type_exp == 'conv' else ''
            print(dir_type)
            save_history = SaveExperiment(root_dir=self.filepath + dir_type)
            name_history = 'train_history_{}_{}_{}'.format(self.state.dataset_name,
                                                        self.model.name,
                                                        self.state.exp)

            print("\nSaving Joined History...")
            if self.state.exp.split('_')[0] == 'k-fold':
                df_history = self.merge_history(self.state.get_history_k(self.state.last_k() - 1))
            else:
                df_history = self.merge_history(self.state.get_history_h())

            save_history.save_history_csv(df_history.to_dict(), name_history)
                

        if ((epoch+1) % 100) == 0:
            print('\nsaving...{} - epochs:{}'.format(self.model.name, epoch+1))
            print(path, '\n')
            # print(self.model.history.history)


            self.state.update_epoch(epoch+1)
            if self.state.exp.split('_')[0] == 'k-fold':
                self.state.update_weights_k(self.state.last_k() - 1, path)
            else:
                self.state.update_weights_h(path)
                
            print('saving weights')
            SaveExperiment(root_dir=self.filepath + 'weights/').save_weights(self.model, name_weights)
            # self.model.save_weights(path)            
            print('saving state...')
            self.state.save_state()

    def merge_history(self, historys):
        merged = pd.read_csv(historys[0])
        for path in historys:
            df = pd.read_csv(path)
            merged = pd.concat([merged, df])
            merged = merged.drop_duplicates()
        
        return merged
        

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

        # checkpoint = ModelCheckpoint(checkpoint_name + '.hdf5', monitor='loss', verbose=1,
                                        # save_best_only=True, mode='auto', save_freq=save_freq)
        # self.callbacks.append(checkpoint)

        self.callbacks.append(CheckpointState(dir_path, state, epochs=epochs))

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

