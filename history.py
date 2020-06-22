import pandas as pd


class History:
    def __init__(self, file_name, history):
        self.file_name = file_name
        self.format = format
        self.history_df = pd.DataFrame(history.history)

    def save(self):
        with open(self.file_name, mode='w') as f:
            self.history_df.to_csv(f)