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

    def save_history_csv(self, history, name):
        path = self.root_dir + name + '.csv'
                
        if not os.path.exists(path.split(name + '.csv')[0]):
            os.makedirs(self.root_dir + '{}/'.format(name.split('_')[1]))

        df = pd.DataFrame(history.history)
        with open(path, mode='w') as f:
            df.to_csv(f)
    
    def save_experiment(self, experiment, name):
        path = self.root_dir + name + '.json'
        if not os.path.exists(path.split(name + '.json')[0]):
            os.makedirs(path.split(name + '.json')[0])

        exp_json = json.dumps(experiment, indent=4)
        with open(path, mode='w') as f:
            f.write(exp_json)