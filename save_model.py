import pandas as pd


class SaveModel:
    def __init__(self,  model, dir_name='./'):
        self.dir_name = dir_name
        self.file_name = model.name
        self.model = model
    
    def save_model(self):
        path = self.dir_name + self.file_name + '.h5'
        self.model.save(path)

    def save_history_csv(self, history):
        df = pd.DataFrame(history.history)
        with open(self.file_name, mode='w') as f:
            df.to_csv(f)