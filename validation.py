from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sh
import numpy as np
from utils import to_categorical

class Holdout:
    def __init__(self, test_size, trainner):
        self.trainner = trainner
        self.test_size = test_size

    def add_score(self, metrics, dict_scores, scores):
        for i, metric in enumerate(metrics):
                dict_scores[metric] = []
                dict_scores[metric].append(scores[i])

    def execute(self, model, inputs, targets):
        dict_scores = {}
        dict_scores['model'] = [model().name]

        print('\n------[executing Hold out {} for {} model]------------------'.format(self.test_size, model.name))
        train_x, test_x, train_y, test_y  = train_test_split(inputs,
                                                            targets, 
                                                            test_size=self.test_size,
                                                            random_state=0, 
                                                            shuffle=True)
        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)

        history = self.trainner.train_model(train_x, train_y, model(), validation_data=(test_x,test_y))

        print('Avaluating model-------------------------------------------------------------')
        scores = model().evaluate(test_x, test_y)

        self.add_score(model().metrics_names, dict_scores, scores)
        

        return dict_scores, history

class KFoldCustom:
    def __init__(self, k, trainner):
        self.k = k
        self.trainner = trainner
    
    def split(self, X):
        n = X.shape[0]
        i = 1
        dataset = np.arange(0, n, dtype=int)
        while i <= self.k:
            idx = np.arange(n * (i - 1) / self.k, n * i / self.k, dtype=int)
            yield np.array(list(set(dataset) - set(idx))), idx
            i += 1
    
    def add_score(self, metrics, dict_score, scores):
        for i, metric in enumerate(metrics):
                dict_score[metric] = []
                dict_score[metric].append(scores[i])

    def execute(self, model, inputs, targets, shuffle = False):
        if shuffle:
            inputs, targets = sh(inputs, targets, random_state=0)
        
        print('\n------[executing {}-fold for {} model]------------------'.format(self.k, model().name))
        scores = {}
        scores['model'] = [model().name]
        n_fold = 1
        historys = []
        for train, test in self.split(inputs):
            print('\n{}-fold'.format(n_fold))
            history = self.trainner.train_model(inputs[train],
                                        to_categorical(targets[train]), 
                                        model(),
                                        validation_data=(inputs[test],targets[test]))

            print('\nAvaluating model-------------------------------------------------------------')
            scores_model = model().evaluate(inputs[test], to_categorical(targets[test]))

            historys.append(history)
            self.add_score(model().metrics_names, scores, scores_model)

            model.resetting_weight()
            n_fold += 1
        
        for metric in {k:scores[k] for k in scores if k != 'model'}:
            scores[metric] = [np.mean(scores[metric])]

        print("Result for the {} model".format(model().name))
        print(scores)

        return scores, historys
