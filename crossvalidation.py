from tensorflow.keras import datasets
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
from utils import to_categorical

class CrossValidation:
    def execute(self):
        print('__execute')


class KFoldValidation(CrossValidation):
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
            targets_train = to_categorical(self.targets[train])
            targets_test = to_categorical(self.targets[test])

            print('\n{}-fold'.format(n_fold))
            self.trainner.train_model(self.inputs[train], targets_train, self.model())
            self.model.resetting_weight()

            print('Avaluating model-------------------------------------------------------------')
            scores = self.model().evaluate(self.inputs[test], targets_test)
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

class Holdout(CrossValidation):
    def __init__(self, model, train_set=None, target_set=None, trainner=None):
        self.inputs = np.concatenate((train_set[0], train_set[1]), axis=0)
        self.targets = np.concatenate((target_set[0], target_set[1]), axis=0)
        self.trainner = trainner
        self.model = model


    def execute(self):
        train_x, test_x, train_y, test_y  = train_test_split(self.inputs, self.targets, test_size=0.2, random_state=0)
        print(train_x.shape, test_x.shape)

        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)

        self.trainner.train_model(train_x, train_y, self.model())
        scores = self.model().evaluate(test_x, test_y)

        results = {}
        for i, metric in enumerate(self.model().metrics_names):
                results[metric] = []
                results[metric].append(scores[i])

        return results
