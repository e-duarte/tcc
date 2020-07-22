import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

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
    
    def save_results(self, results, name_experiment):
        path = self.dir_name + 'experiment_'+ name_experiment+ '_' + self.get_date() + '.csv'
        df = pd.DataFrame(results)
        with open(path, mode='w') as f:
            df.to_csv(f)

    def save_history_csv(self, history, name):
        path = self.dir_name + '{}/'.format(name.split('_')[1]) + name + '_' + self.get_date() + '.csv'
        df = pd.DataFrame(history.history)
        with open(path, mode='w') as f:
            df.to_csv(f)
    def save_roc_curve(self, fpr, tpr, auc, name):
        path = self.dir_name + '{}/'.format(name.split('_')[1]) + name + '_' + self.get_date() + '.png'

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(path, format='png')