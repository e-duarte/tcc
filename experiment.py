from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from dataset import DatasetFactory
from optimizers import Optimizers
from models import FactoryModel
from training_models import Trainner
from validation import ValidationFactory
from save import SaveExperiment
from utils import PlotGraph
from metrics import f1_score
import json, os

class Experiment:
    def __init__(self, experiment, dataset):
        self.experiment = experiment
        self.dataset = dataset
        self.state = None

        if experiment['load_state']:
            print('\n\nLoad experiment state...\n\n')
            self.state = ExperimentState(load=True, state_url=experiment['state_url'])
            self.dataset.restore_state(self.state.dataset)
        else:
            print('\n\nMaking new experiment state..\n\n')
            exps = {'k-fold':[], 'holdout':[]}
            for exp in experiment['exp']:
                exps[exp] = experiment['exp'][exp]
                        
            # k, h = experiment['exp']['k-fold'], experiment['exp']['holdout']
            self.dataset.shuffle()

            model_name = self.models_name()
            experiment_name = self.experiment_name()
            self.state = ExperimentState(
                dataset=self.dataset.dataset[2], 
                k=exps['k-fold'],
                h=exps['holdout'], 
                experiment_url=experiment['dir'], 
                dataset_name=experiment['dataset'],
                type_exp=experiment['type'],
                load=experiment['load_state'],
                epochs_total=experiment['epochs'],
                state_url=experiment['state_url'].format(experiment['dataset'] + '_' + model_name + '_' + experiment_name)
            )

            # print(self.state.to_dict())

        #compiler parameters
        optimizer = experiment['optimizer']
        opt_params = experiment['opt_params']
        loss = experiment['loss']
        metrics = [m for m in experiment['metrics'] if m != 'f1_score']
        metrics.append(f1_score)

        self.compiler_params= dict([
            ('optimizer', Optimizers(optimizer, opt_params).optimizer()),
            ('loss', loss),
            ('metrics', metrics)
        ])
        
        #Config training
        callbacks = []
        datagen = None
        if experiment['data_augmentation']:
            datagen = ImageDataGenerator(width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        horizontal_flip=True)
        if experiment['decay']:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, min_lr=0.1e-3))
        
        epochs = experiment['epochs'] - self.state.epochs if experiment['load_state'] else experiment['epochs']
        self.trainner = Trainner(
            epochs=epochs,
            batch_size=experiment['batch'], 
            data_augmentation=datagen, 
            callbacks=callbacks, 
            dir_path=experiment['dir'],
            state=self.state
        )

    def execute(self):
        for model in self.experiment['models']:
            for exp in self.experiment['exp']:
                print('===================Applying {}====================================='.format(exp))
                for x in self.state.get_exp(exp):
                    x = x/100 if exp == 'holdout' else x
                    
                    validator = ValidationFactory(name=exp, x=x, trainner=self.trainner, state=self.state).validator
                    dict_scores = validator.execute(
                        inputs=self.dataset.dataset[0],
                        targets=self.dataset.dataset[1],
                        config_model={
                            'name':model,
                            'size':self.dataset.shape,
                            'params':self.compiler_params,
                            'init':self.experiment['initializers']
                        }
                    )

                    print('Resetting experiment...')
                    self.state.epochs = 0
                    self.state.save_state()
                    self.trainner.epochs = self.experiment['epochs']


                    self.save_results(dict_scores, model, '{}{}'.format(exp, x))
        self.save_experiment()
    
    def experiment_name(self):
        name_exp = ''
        for exp in self.experiment['exp']:
            name_exp += exp
            for x in self.experiment['exp'][exp]:
                name_exp += str(x)
        return name_exp

    def models_name(self):
        name_models = ''
        for model in self.experiment['models']:
            name_models += model
        return name_models

    def save_experiment(self):
        print('saving experiment...')
        name_exp = ''
        for exp in self.experiment['exp']:
            name_exp += exp
            for x in self.experiment['exp'][exp]:
                name_exp += str(x)
        
        name_models = ''
        for model in self.experiment['models']:
            name_models += model
        
        SaveExperiment(self.experiment['dir'] + 'results/{}/'.format(self.experiment['type'])).save_experiment(self.experiment,
                                                                            'exp_{}_{}_{}'.format(
                                                                                self.experiment['dataset'],
                                                                                name_models,
                                                                                name_exp))

    def save_results(self, dict_scores, model_name, validation):
        results_path = self.experiment['dir'] + 'results/{}/{}/'.format(self.experiment['type'], model_name)
        save = SaveExperiment(results_path)
        print('========================= [Saving models results] ===================================')
        
        print('Saving results of test...')
        save.save_results(
            dict_scores['scores'],
            'test_{}_{}_{}'.format(
                self.experiment['dataset'],
                model_name,
                validation
            )
        )

        print('Saving confusion matrix of test...')
        plot = PlotGraph(self.dataset.classes_names, self.dataset.classes_names, path=results_path, save=True)
        plot.plot_cm(
            dict_scores['cm'],
            'test_cm_{}_{}_{}.png'.format(
                self.experiment['dataset'],
                model_name,
                validation
            )
        )

        print('Saving ROC Curve of test...')
        plot = PlotGraph(path=results_path, save=True)
        (fpr, tpr, auc) = dict_scores['roc']
        plot.plot_roc(
            fpr,
            tpr,
            auc,
            'test_roc_{}_{}_{}.png'.format(
                self.experiment['dataset'],
                model_name,
                validation
            )
        )

class ExperimentState:
    def __init__(self, dataset=[], k=None, h=None, experiment_url=None, load=False, state_url=None, dataset_name=None, type_exp=None, epochs_total=None):
        if not load:
            self.dataset = dataset
            self.dataset_name = dataset_name
            self.experiment_url = experiment_url
            self.epochs = 0
            range_k = k[0] if k != [] else 0
            self.history_k = [[] for i in range(range_k)]
            self.history_h = []
            self.weights_k = [[] for i in range(range_k)]
            self.weights_h = []
            self.k = [1 for i in range(len(k))]
            self.h = h
            self.state_url = state_url
            self.exp_k = k
            self.load = load
            self.exp = ''
            self.epochs_total = epochs_total
            self.type_exp = type_exp
        else:
            state = self.load_json(state_url)
            self.dataset = state['dataset']
            self.dataset_name = state['dataset_name']
            self.experiment_url = state['experiment_url']
            self.epochs = state['epochs']
            self.history_k = state['history_k']
            self.history_h = state['history_h']
            self.weights_k = state['weights_k']
            self.weights_h = state['weights_h']
            self.k = state['k']
            self.h = state['h']
            self.state_url = state['state_url']
            self.exp_k = state['exp_k']
            self.load = state['load']
            self.exp = state['exp']
            self.epochs_total = state['epochs_total']
            self.type_exp = state['type']
    
    def load_json(self, url):
        state = {}
        with open(url, mode='r') as f:
            state = json.loads(f.read())
        return state

    def update_epoch(self, epochs):
        self.epochs = epochs
    
    def update_weights_k(self, k, url):
        self.weights_k[k].append(url)

    def update_history_k(self, k, url):
        self.history_k[k].append(url)
    
    def update_weights_h(self, url):
        self.weights_h.append(url)

    def update_history_h(self, url):
        self.history_h.append(url)

    def update_h(self):
        self.weights_h = []
        self.history_h = []
        del(self.h[0])

    def update_k(self, k):
        self.k[k] += 1
    
    def reset_weighs_heights_k(self):
        self.weights_k = [[] for i in range(len(self.exp_k))]
        self.history_k = [[] for i in range(len(self.exp_k))]
            
    def get_exp(self, name):
        if name  == 'k-fold':
            # return self.last_exp_k()
            return self.exp_k
        elif name == 'holdout':
            return [i for i in self.h]

    def last_exp_h(self):
        return self.h[0]
    
    def last_exp_k(self):
        exp_k = []
        # k = 0
        for i, j in zip(self.k, self.exp_k):
            if i <= j:
                exp_k = j
                k = i
                break
        return exp_k
    
    def last_k(self):
        # exp_k = []
        k = 0
        for i, j in zip(self.k, self.exp_k):
            if i <= j:
                # exp_k = j
                k = i
                break
            else:
                k = i
        return k
    
    def get_weights_h(self):
        weights = []

        if not self.weights_h == []:
            weights = self.weights_h[-1]
        return weights
    
    def get_weights_k(self, k):
        weights = []

        if not self.weights_k[k] == []:
            weights = self.weights_k[k][-1]

        return weights

    def get_history_h(self):
        return self.history_h
    
    def get_history_k(self, k):
        return self.history_k[k]
    
    def active_experiment(self):
        if self.weights_k != [[]]:
            return 'kfold_{}'.format(self.last_exp_k())
        elif self.weights_h != []:
            return 'holdout_{}'.format(self.last_exp_h())
    
    def save_state(self):
        path = ''
        for p in self.state_url.split('/')[:-1]:
            path += p + '/'
        print(path)
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        state = self.to_dict()
        with open(self.state_url, mode='w') as f:
            json.dump(state, f, indent=4)
    
    def to_dict(self):
        state = {
            'dataset' : self.dataset,
            'dataset_name': self.dataset_name,
            'experiment_url' : self.experiment_url,
            'epochs' : self.epochs,
            'history_k' : self.history_k,
            'history_h' : self.history_h,
            'weights_k' : self.weights_k,
            'weights_h' : self.weights_h, 
            'k' : self.k,
            'h' : self.h,
            'state_url' : self.state_url,
            'exp_k' : self.exp_k,
            'load': self.load,
            'exp': self.exp,
            'epochs_total': self.epochs_total,
            'type': self.type_exp
        }
        return state

if __name__ == '__main__':
    exp = {}
    with open('./experiment.json', mode='r') as f:
        exp = json.loads(f.read())

    exp_estate = ExperimentState(
        dataset=exp['dataset'],
        k=exp['exp']['k-fold'],
        experiment_url=exp['dir'] + 'exp_test.json',
        state_url='./state_test.json',
        h=exp['exp']['holdout']
    )
    for i in range(10):
        exp_estate.update_k(0)

    exp_estate.update_h()
    print(exp_estate.last_k())
    exp_estate.save_state()

