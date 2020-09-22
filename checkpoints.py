from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from save import SaveExperiment
from tensorflow import keras
import pandas as pd
import os, time

class WeightsCheckpoint(keras.callbacks.Callback):
    def __init__(self, url, state):
        self.state = state
        self.save_weights = SaveExperiment(root_dir= url + 'weights/')
        self.save_state = SaveExperiment(root_dir= url + 'states/')
        
    def on_epoch_end(self, epoch, logs=None):
        valid_name = self.model.name.split('_')[2].split('-')[0]
        valid_x = self.model.name.split('_')[2].split('-')[1]

        valid_h = float(valid_x)*100 if valid_name == 'holdout' else None
        valid_k = int(valid_x) if valid_name == 'kfold' else None

        state_validation = self.state.get_state_validation(valid_name=valid_name, k=valid_k, h=valid_h)

        if ((epoch + 1) % 4) == 0:
            print('\nSAVING MODEL WEIGHTS...')

            if valid_name == 'holdout':
                state_validation.epochs = epoch + 1
            elif valid_name == 'kfold':
                state_validation.epochs[state_validation.current_k - 1] = epoch + 1
            
            self.save_weights.save_weights(self.model, self.model.name)
            self.save_state.save_state(self.state.to_dict())
            

class HistoryCheckpoint(keras.callbacks.Callback):
    def __init__(self, url, path_tmp, state, metrics):
        self.state = state
        self.path_tmp = path_tmp
        self.url = url + 'results/{}/'
        self.history = {m:[] for m in metrics}

    def to_reset_history(self):
        for m in self.history:
            self.history[m] = []
    
    def on_epoch_end(self, epoch, logs=None):
        for m in logs:
            self.history[m].append(logs[m])
        
        url = self.url.format(self.model.name.split('_')[1])
        self.save_history = SaveExperiment(root_dir=url)

        valid_name = self.model.name.split('_')[2].split('-')[0]
        valid_x = self.model.name.split('_')[2].split('-')[1]

        valid_h = float(valid_x)*100 if valid_name == 'holdout' else None
        valid_k = int(valid_x) if valid_name == 'kfold' else None

        state_validation = self.state.get_state_validation(valid_name=valid_name, k=valid_k, h=valid_h)

        if ((epoch + 1) % 4) == 0:
            print('\nSAVING HISTORY...')

            tmp_history = pd.DataFrame(self.history)
            path_history = ''

            if valid_name == 'holdout':
                path_history =  state_validation.history
            elif valid_name == 'kfold':
                path_history = state_validation.historys[state_validation.current_k - 1]

            history = pd.DataFrame(columns=tmp_history.columns)

            if os.path.exists(path_history):
                history = pd.read_csv(path_history, index_col=[0])
            
            cp_history = history.copy()
            join_history = cp_history.append(tmp_history, ignore_index=True)

            self.save_history.save_history_csv(join_history, 'history_' + self.model.name)
            self.to_reset_history()
