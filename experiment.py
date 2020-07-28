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

class Experiment:
    def __init__(self, experiment, dataset):
        self.experiment = experiment
        self.dataset = dataset

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
        
        self.trainner = Trainner(
            epochs=experiment['epochs'],
            batch_size=experiment['batch'], 
            data_augmentation=datagen, 
            callbacks=callbacks, 
            dir_path=experiment['dir']
        )

    def execute(self):
        for model in self.experiment['models']:
            for exp in self.experiment['exp']:
                print('===================Applying {}====================================='.format(exp))
                for x in self.experiment['exp'][exp]:
                    x = x/100 if exp == 'holdout' else x
                    
                    validator = ValidationFactory(name=exp, x=x, trainner=self.trainner).validator
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

                    self.save_results(dict_scores, model, '{}{}'.format(exp, x))
        self.save_experiment()
    
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
                                                                            'exp{}_{}_{}'.format(
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

        print('Saving history of train...')
        for i, history in enumerate(dict_scores['history']):
            save.save_history_csv(
                history,
                'train_history_{}_{}_{}_{}'.format(
                    self.experiment['dataset'],
                    model_name,
                    validation,
                    i
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