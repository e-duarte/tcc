from tensorflow.keras import datasets
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

class KFoldValidation:
    def __init__(self, model, train_set=None, target_set=None, k=5, trainner=None):
        self.inputs = np.concatenate((train_set[0], train_set[1]), axis=0)
        self.targets = np.concatenate((target_set[0], target_set[1]), axis=0)
        self.kfold = StratifiedKFold(n_splits=k, shuffle=True)
        self.trainner = trainner
        self.model = model

    def execute(self):
        results = {}
        n_fold = 1
        print('\n------[executing k-fold for {} model]------------------'.format(self.model.name))
        for train, test in self.kfold.split(self.inputs, self.targets):
            print('\n{}-fold'.format(n_fold))

            train_labels, _ = to_categorical(self.targets[train], None)
            self.trainner.train_model(self.inputs[train], self.targets[train], self.model())
            self.model.resetting_weight()
            print('Avaluating model-------------------------------------------------------------')
            scores = self.model().evaluate(self.inputs[test], self.targets[test])
            for i, metric in enumerate(self.model().metrics_names):
                results[metric] = []
                results[metric].append(scores[i])
            n_fold += 1
        
        for metric in results:
            results[metric] = [np.mean(results[metric])]
        results['model'] = [self.model.name]

        print("Result for the {} model".format(self.model.name))
        print(results)
        return results

