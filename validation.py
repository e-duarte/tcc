from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sh
from sklearn.metrics import confusion_matrix
import numpy as np
from utils import to_categorical
from metrics import get_roc_curve
from models import FactoryModel
# from pandas_ml import ConfusionMatrix

class Holdout:
    def __init__(self, test_size, trainner):
        self.trainner = trainner
        self.test_size = test_size

    def add_score(self, metrics, dict_scores, scores):
        for i, metric in enumerate(metrics):
                dict_scores[metric] = []
                dict_scores[metric].append(scores[i])

    def execute(self, inputs, targets, config_model=None):
        model = FactoryModel(
                config_model['name'],
                config_model['name']+ '_split{}'.format(self.test_size),
                config_model['size'],
                config_model['params']).get_model()

        dict_scores = {}
        dict_scores['scores'] = {}
        dict_scores['scores']['model'] = [model().name]

        print('\n------[executing Hold out {} for {} model]------------------'.format(self.test_size, model.name))
        train_x, test_x, train_y, test_y  = train_test_split(inputs,
                                                            targets, 
                                                            test_size=self.test_size,
                                                            random_state=0, 
                                                            shuffle=False)
        # train_y = to_categorical(train_y)
        # test_y = to_categorical(test_y)

        history = self.trainner.train_model(train_x, to_categorical(train_y), model(), validation_data=(test_x,to_categorical(test_y)))

        print('Avaluating model-------------------------------------------------------------')
        scores = model().evaluate(test_x, to_categorical(test_y))

        self.add_score(model().metrics_names, dict_scores['scores'], scores)


        (fpr, tpr, auc) = get_roc_curve(to_categorical(test_y), model().predict(test_x))
        dict_scores['roc'] = (fpr, tpr, auc)
        dict_scores['history'] = history
        dict_scores['cm'] = confusion_matrix(test_y, np.argmax(model().predict(test_x), axis=1))
        # print(test_y, np.argmax(model().predict(test_x), axis=1))
        # dict_scores['cm'] = ConfusionMatrix(test_y, np.argmax(model().predict(test_x), axis=1))

        print("Result for the {} model".format(model().name))
        print(dict_scores['scores'])
        
        return dict_scores

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

    def execute(self, inputs, targets, shuffle=False, config_model=None):
        if shuffle:
            inputs, targets = sh(inputs, targets, random_state=0)
        
        print('\n------[executing {}-fold for {} model]------------------'.format(self.k, config_model['name']))
        scores_dict = {}
        scores_dict['scores'] = {}
        scores_dict['scores']['model'] = [config_model['name']]
        fprs = []
        tprs = []
        aucs = []
        scores = []
        historys = []
        cms = []

        n_fold = 1
        for train, test in self.split(inputs):
            print('\n{}-fold'.format(n_fold))
            model = FactoryModel(
                config_model['name'],
                config_model['name']+ '_k{}'.format(n_fold),
                config_model['size'],
                config_model['params']).get_model()
            
            history = self.trainner.train_model(inputs[train],
                                        to_categorical(targets[train]), 
                                        model(),
                                        validation_data=(inputs[test], to_categorical(targets[test])))

            print('\nAvaluating model-------------------------------------------------------------')

            scores_model = model().evaluate(inputs[test], to_categorical(targets[test]))

            #roc curve and auc
            (fpr, tpr, auc) = get_roc_curve(to_categorical(targets[test]), model().predict(inputs[test]))

            #get confunsion matrix
            cms.append(
                confusion_matrix(
                    targets[test], 
                    np.argmax(model().predict(inputs[test]), axis=1)
                )
            )

            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(auc)
            
            historys.append(history)
            scores.append(scores_model)

            model.resetting_weight()
            n_fold += 1
        
        scores = np.array(scores)
        for i, m in enumerate(model().metrics_names):
            scores_dict['scores'][m] = [scores[:,i].mean()]

        roc_max = (fprs[0], tprs[0], aucs[0])

        for i in range(len(aucs)):
            if roc_max[2] > aucs[i]:
                roc_max = (fprs[i], tprs[i], aucs[i])

        #mean confusion matrix
        cm_mean = np.zeros(cms[0].shape)
        for cm in cms:
            cm_mean += (cm/cms[0].shape[0]).astype('int32')

        cm_mean = cm_mean.astype('int32')
        
        scores_dict['roc'] = roc_max
        scores_dict['history'] = historys
        scores_dict['cm'] = cm_mean
        
        print("Result for the {} model".format(model().name))
        print(scores_dict['scores'])

        return scores_dict
