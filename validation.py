from tensorflow.keras import datasets
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
import numpy as np
from utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

class CrossValidation:
    def execute(self):
        print('__execute')


class KFoldValidation(CrossValidation):
    def __init__(self, model, train_set=None, test_set=None, k=5, trainner=None):
        self.inputs = np.concatenate((train_set[0], test_set[0]), axis=0)
        self.targets = np.concatenate((train_set[1], test_set[1]), axis=0)
        # self.train_x, self.test_x, self.train_y, self.test_y  = train_test_split(self.inputs, self.targets, test_size=0.1, random_state=0, shuffle=True)

        # self.train = train_set
        # self.test = test_set
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
            # targets_test_1 = to_categorical(self.test_y)

            print('\n{}-fold'.format(n_fold))
            self.trainner.train_model(self.inputs[train], targets_train, self.model())
            self.model.resetting_weight()

            print('Avaluating model-------------------------------------------------------------')
            scores = self.model().evaluate(self.inputs[test], targets_test)
            # print('Testing model-------------------------------------------------------------')
            # scores = self.model().evaluate(self.test_x, targets_test_1)

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
        print('\n------[executing Hold out for {} model]------------------'.format(self.model.name))
        train_x, test_x, train_y, test_y  = train_test_split(self.inputs, self.targets, test_size=0.1, random_state=0, shuffle=True)
        print(train_x.shape, test_x.shape)

        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)

        self.trainner.train_model(train_x, train_y, self.model())

        print('Avaluating model-------------------------------------------------------------')
        scores = self.model().evaluate(test_x, test_y)

        results = {}
        for i, metric in enumerate(self.model().metrics_names):
                results[metric] = []
                results[metric].append(scores[i])
        results['model'] = [self.model.name]

        return results


def cross_validation(model, X, y, epochs, batch_size, cv, callbacks=[]):
    def get_model(model):
        return model
    
    model = KerasClassifier(get_model, epochs=epochs, batch_size=batch_size)
    scores = cross_val_score(model, X, y, cv=cv)

    return scores.mean()
    