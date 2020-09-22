from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sh
from sklearn.metrics import confusion_matrix
from utils import to_categorical
from metrics import get_roc_curve
from models import FactoryModel
import numpy as np

class ValidationFactory:
    def __init__(self, name, trainner, x, state=None):
        self.validator = None
        if name == 'kfold':
            self.validator = KFoldCustom(k=x, trainner=trainner, state=state)
        elif name == 'holdout':
            self.validator = Holdout(test_size=x, trainner=trainner, state=state)
    
class Holdout:
    def __init__(self, test_size, trainner, state=None):
        self.trainner = trainner
        self.test_size = test_size
        self.state = state.get_state_validation(valid_name='holdout', h=test_size*100)

    def add_score(self, metrics, dict_scores, scores):
        for i, metric in enumerate(metrics):
                dict_scores[metric] = []
                dict_scores[metric].append(scores[i])

    def execute(self, inputs, targets, config_model=None, dataset_name=''):
        model = FactoryModel(
                config_model['name'],
                '{}_{}_holdout-{}'.format(dataset_name, config_model['name'], self.test_size),
                config_model['size'],
                config_model['params'],
                config_model['init']).get_model()

        if self.state.epochs > 0:
            model().load_weights(self.state.weights)
        else:
            #initialize weight and history path for the model
            self.state.weights += model().name + '.h5'
            self.state.history += '{}/history_{}.csv'.format(config_model['name'], model().name)

        dict_scores = {}
        dict_scores['scores'] = {}
        dict_scores['scores']['model'] = [model().name]

        print('\n------[executing Hold out {} for {} model]------------------'.format(self.test_size*100, model().name))
        train_x, test_x, train_y, test_y  = train_test_split(inputs,
                                                            targets, 
                                                            test_size=self.test_size,
                                                            random_state=0, 
                                                            shuffle=False)
        history = self.trainner.train_model(train_x,
                                            to_categorical(train_y),
                                            model(),
                                            validation_data=(test_x,to_categorical(test_y)),
                                            init_epoch=self.state.epochs
                                            )

        print('Avaluating model-------------------------------------------------------------')
        scores = model().evaluate(test_x, to_categorical(test_y))

        self.add_score(model().metrics_names, dict_scores['scores'], scores)


        (fpr, tpr, auc) = get_roc_curve(to_categorical(test_y), model().predict(test_x))
        dict_scores['roc'] = (fpr, tpr, auc)
        dict_scores['history'] = [history]
        dict_scores['cm'] = confusion_matrix(test_y, np.argmax(model().predict(test_x), axis=1))


        print("Result for the {} model".format(model().name))
        print(dict_scores['scores'])

        self.state.status = True
        
        return dict_scores

class KFoldCustom:
    def __init__(self, k, trainner, state=None):
        self.k = k
        self.trainner = trainner
        self.state = state.get_state_validation(valid_name='kfold', k=k)
        self.state_i = self.state.current_k
    
    def split(self, X):
        n = X.shape[0]
        dataset = np.arange(0, n, dtype=int)
        while self.state_i <= self.k:
            idx = np.arange(n * (self.state_i - 1) / self.k, n * self.state_i / self.k, dtype=int)
            yield np.array(list(set(dataset) - set(idx))), idx
            self.state_i += 1

    def execute(self, inputs, targets, config_model=None, dataset_name=''):
        
        print('\n------[executing {}-fold for {} model]------------------'.format(self.k, config_model['name']))
        scores_dict = {}
        scores_dict['scores'] = {}
        scores_dict['scores']['model'] = [config_model['name']]
        fprs = []
        tprs = []
        aucs = []
        scores = []
        cms = []

        #training
        n_fold = self.state_i
        for train, test in self.split(inputs):
            print('\n{}-fold'.format(n_fold))
            model = FactoryModel(
                config_model['name'],
                '{}_{}_kfold-{}_{}'.format(dataset_name, config_model['name'], self.k, n_fold),
                config_model['size'],
                config_model['params']).get_model()
            
            
            if self.state.get_epochs(self.state_i - 1) > 0:
                model().load_weights(self.state.get_weights(self.state_i - 1))
            else:
                #initialize weight and history path for the model i
                self.state.weights[self.state_i - 1] += model().name + '.h5'
                self.state.historys[self.state_i - 1] += '{}/history_{}.csv'.format(config_model['name'], model().name)


            self.trainner.train_model(inputs[train],
                                    to_categorical(targets[train]), 
                                    model(),
                                    validation_data=(inputs[test], to_categorical(targets[test])),
                                    init_epoch=self.state.get_epochs(self.state_i - 1),
                                    )
            n_fold += 1
            self.state.current_k = n_fold

        self.state_i = 1
        n_fold = self.state_i
        for train, test in self.split(inputs):
            print('\nAvaluating {}-fold model\n'.format(n_fold))
            model = FactoryModel(
                config_model['name'],
                config_model['name']+ '_k{}'.format(n_fold),
                config_model['size'],
                config_model['params']).get_model()

            model().load_weights(self.state.get_weights(self.state_i-1))

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
            
            # historys.append(history)
            scores.append(scores_model)
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
        # scores_dict['history'] = historys
        scores_dict['cm'] = cm_mean
        
        print("Result for the {} model".format(model().name))
        print(scores_dict['scores'])

        self.state.statu = True

        return scores_dict

if __name__ == '__main__':
    kfold = KFoldCustom(3, None)

    for train, test in kfold.split(np.array([1,2,3,4,5,6])):
        print(train, test)

