import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
import json

class SaveExperiment:
    def __init__(self, root_dir='./'):
        self.root_dir = root_dir

        if not os.path.exists(root_dir):
            print('Creating Diretory...')
            os.makedirs(root_dir)
    
    def save_model(self, model, name):
        path = self.root_dir + name + '.h5'
        model.save(path)
    
    def save_results(self, results, name_experiment):
        path = self.root_dir + name_experiment + '.csv'

        print(path.split(name_experiment + '.csv')[0])
        if not os.path.exists(path.split(name_experiment + '.csv')[0]):
            os.makedirs(path.split(name_experiment + '.csv')[0])

        df = pd.DataFrame(results)
        with open(path, mode='w') as f:
            df.to_csv(f)

    def save_history_csv(self, df_history, name):
        path = self.root_dir + name + '.csv'
                
        if not os.path.exists(path.split(name + '.csv')[0]):
            os.makedirs(self.root_dir + '{}/'.format(name.split('_')[1]))

        # df = pd.DataFrame(history)
        with open(path, mode='w') as f:
            df_history.to_csv(f)
    
    def save_experiment(self, experiment, name):
        path = self.root_dir + name + '.json'
        if not os.path.exists(path.split(name + '.json')[0]):
            os.makedirs(path.split(name + '.json')[0])

        exp_json = json.dumps(experiment, indent=4)
        with open(path, mode='w') as f:
            f.write(exp_json)
    
    def save_weights(self, model, name):
        path = self.root_dir + name + '.h5'

        print('\n', path)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        
        model.save_weights(path)
    
    def save_state(self, state):
        path = self.root_dir + 'state_{}.json'.format(state['id_exp'])
        with open(path, mode='w') as f:
            json.dump(state, f, indent=4)