import pandas as pd
from datetime import datetime

class SaveModel:
    def __init__(self,  model=None, dir_name='./'):
        self.dir_name = dir_name
        self.file_name = model.name if model else '' 
        self.model = model
    
    def get_date(self):
        now = datetime.now()
        return '{}_{}_{}'.format(now.day, now.month, now.year)

    def save_model(self):
        path = self.dir_name + self.file_name + self.get_date() + '.h5'
        self.model.save(path)
    
    def save_results(self, results):
        path = self.dir_name + 'experiment' + self.get_date() + '.csv'
        df = pd.DataFrame(results)
        with open(path, mode='w') as f:
            df.to_csv(f)

    def save_history_csv(self, history, name):
        path = self.dir_name + name + '_' + self.get_date() + '.csv'
        df = pd.DataFrame(history.history)
        with open(path, mode='w') as f:
            df.to_csv(f)