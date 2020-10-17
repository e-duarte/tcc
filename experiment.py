from tensorflow.keras.preprocessing.image import ImageDataGenerator
from checkpoints import WeightsCheckpoint, HistoryCheckpoint, CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau
from dataset import DatasetFactory
from optimizers import Optimizers
from models import FactoryModel
from training_models import Trainner
from validation import ValidationFactory
from save import SaveExperiment
from utils import PlotGraph
from metrics import f1_score
from state import LoaderState
import json, os

class Experiment:
    def __init__(self, experiment):
        self.experiment = experiment

        processing = experiment['processing']

        #config dataset
        self.dataset = DatasetFactory(
            name=experiment['dataset'],
            flat=processing['flat'],
            concat=processing['concat'],
            expand=processing['expand'],
            normalize=processing['normalize'],
        )
        
        #config state of the experiment
        self.state = LoaderState(
            id_exp=self.experiment_name(),
            epochs=experiment['epochs'],
            dataset=self.dataset,
            valid_exp=experiment['exp'],
            url=experiment['dir']
        ).state


        #compiler parameters
        optimizer = experiment['optimizer']
        opt_params = experiment['opt_params']
        loss = experiment['loss']
        metrics = [m for m in experiment['metrics'] if m != 'f1_score']
        history_metrics = [m.lower() for m in experiment['metrics'] if m != 'f1_score']
        metrics.append(f1_score)

        self.compiler_params = dict([
            ('optimizer', Optimizers(optimizer, opt_params).optimizer()),
            ('loss', loss),
            ('metrics', metrics)
        ])

        #Config training
        callbacks = []

        history_metrics.insert(0, 'loss')
        history_metrics.append('f1_score')

        cp = [m for m in history_metrics]

        for m in cp:
            history_metrics.append('val_' + m)

        callbacks.append(HistoryCheckpoint(
            self.experiment['dir'],
            self.state,
            history_metrics))
            
        callbacks.append(WeightsCheckpoint(self.experiment['dir'], self.state))

        
        
        if experiment['decay']:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, min_lr=0.1e-3))

        datagen = None
        if experiment['data_augmentation']:
            datagen = ImageDataGenerator(width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        horizontal_flip=True)
        
        self.trainner = Trainner(
            epochs=experiment['epochs'],
            batch_size=experiment['batch'], 
            data_augmentation=datagen, 
            callbacks=callbacks, 
            dir_path=experiment['dir'],
            state=self.state
        )

    def execute(self):
        if self.state.status:
            print('[INFO]THE EXPERIMENT HAS ENDED')
            exit()

        for model in self.experiment['models']:
            for exp in self.experiment['exp']:
                print('===================Applying {}====================================='.format(exp))
                valid_exps = self.state.get_validation(valid_name=exp)
                print('experimentos para {}'.format(exp))
                print(valid_exps)
                for x in valid_exps:
                    validator = ValidationFactory(name=exp, x=x, trainner=self.trainner, state=self.state).validator
                    dict_scores = validator.execute(
                        inputs=self.dataset.dataset[0],
                        targets=self.dataset.dataset[1],
                        config_model={
                            'name':model,
                            'size':self.dataset.shape,
                            'params':self.compiler_params,
                            'init':self.experiment['initializers']
                        },
                        dataset_name=self.experiment['dataset']
                    )

                    SaveExperiment(
                        root_dir=self.experiment['dir'] + 'states/'
                    ).save_state(self.state.to_dict())

                    self.save_results(dict_scores, model, '{}{}'.format(exp, x))
        
        self.state.status = True
        SaveExperiment(
            root_dir=self.experiment['dir'] + 'states/'
        ).save_state(self.state.to_dict())
        self.save_experiment()
        print('[INFO]THE EXPERIMENT HAS ENDED')
    
    def experiment_name(self):
        name_exp = ''
        name_exp += self.models_name()
        name_exp += self.experiment['dataset']
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
        SaveExperiment(self.experiment['dir'] + 'results/').save_experiment(self.experiment, self.experiment_name())

    def save_results(self, dict_scores, model_name, validation):
        results_path = self.experiment['dir'] + 'results/{}/'.format(model_name)
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

