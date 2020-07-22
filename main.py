import numpy as np
import pandas as pd
from metrics import f1_score
from save_model import SaveModel
from optimizers import Optimizers
from parameters import params_exp
from training_models import Trainner
from tensorflow.keras import datasets
from validation import Holdout, KFoldCustom
from models import Alexnet, Resnet34, DeepAutoencoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import preprocessing, expand_dims, vetorizar_data, to_categorical, PlotGraph
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import json as Json

models_names = params_exp['models']
optimizer = params_exp['optimizer']
opt_params = params_exp['opt_params']
loss = params_exp['loss']
metrics = [m for m in params_exp['metrics'] if m != 'f1_score']
initializers = params_exp['initializers']
epochs = params_exp['epochs']
batch = params_exp['batch_size']
load_model = params_exp['load_model']
data_augmentation = params_exp['data_augmentation']
decay = params_exp['decay_rate']
type_exp = params_exp['type']
dir_save = params_exp['dir_save']
# cross = params_exp['cross']
k = params_exp['k-fold']
h = params_exp['holdout']
dataset_name =  params_exp['dataset']


def get_dataset(name):
    if 'mnist' == name:
        return datasets.mnist.load_data()
    elif 'cifar10' == name:
        return datasets.cifar10.load_data()

(train_images, train_labels), (test_images, test_labels) = get_dataset(dataset_name)

def get_classes_names():
    classes_names = [i for i in range(10)] if dataset_name == 'mnist' else ['airplane', 'automobile',
                                                                            'bird', 'cat', 'deer',
                                                                            'dog', 'frog', 'horse', 
                                                                            'ship', 'truck']
    return classes_names

def concat_dict(dicts):
    if not dicts:
        return []
    dict_join = {}

    for key in dicts[0]:
            dict_join[key] = []

    for d in dicts:
        for key in d:
            dict_join[key] = dict_join[key] + d[key]
    
    return dict_join
    
def build_and_compile_model(model_name, initializers, size, params_compile):
    model = None
    if model_name == 'alexnet':
        model = Alexnet(input_shape=size, initializers=initializers, name=model_name)
    elif model_name == 'resnet':
        model = Resnet34(input_shape=size, name=model_name)
    elif model_name == 'autoencoder':
        model = DeepAutoencoder(name=model_name)

    model.compile(params_compile)
    return model

def initialize_models():
    models = []
    metrics.append(f1_score)
    params_compile = dict([
        ('optimizer', Optimizers(optimizer, opt_params).optimizer()),
        ('loss', loss),
        ('metrics', metrics)
    ])

    dims = train_images.shape[1:] if 'mnist' != dataset_name else train_images.shape[1:] + (1,)
    
    for name in models_names:
        models.append(build_and_compile_model(name, initializers,dims, params_compile))

    for model in models:
        model().summary()
    
    return models

def preprocessing_data():
    global train_images, test_images
    global train_labels, test_labels
    # preprocessing(train_images, test_images)
    if type_exp == 'conv':    
        train_images, test_images = expand_dims(train_images, test_images)
    else:
        train_images, test_images = vetorizar_data(train_images, test_images)

def training():
    callbacks = []
    datagen = None
    if params_exp['data_augmentation']:
        datagen = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True)
    if decay:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, min_lr=0.1e-3))
                                    
    return Trainner(epochs=epochs,batch_size=batch, data_augmentation=datagen, callbacks=callbacks)

def experiment():
    kfold_exp = []
    holdout_exp = []

    inputs = np.concatenate((train_images, test_images), axis=0)
    targets = np.concatenate((train_labels, test_labels), axis=0)

    if len(k) != 0:
        print('\nApply K-fold==========================================================================')
        
        for i in k:
            scores_models = []
            historys_models = []
            roc_curve_models = []
            cm_models = []

            for model in models:
                model.resetting_weight()
                kfold = KFoldCustom(k=i, trainner=trainner)
                dict_scores = kfold.execute(model, inputs, targets, shuffle=True)
                scores_models.append(dict_scores['scores'])
                historys_models.append(dict_scores['history'])
                roc_curve_models.append(dict_scores['roc'])
                cm_models.append(dict_scores['cm'])
            kfold_exp.append((concat_dict(scores_models), historys_models, roc_curve_models, cm_models))
    
    if len(h) != 0:
        print('\nApply Hold Out=========================================================================')
        
        for i in h:
            scores_models = []
            historys_models = []
            roc_curve_models = []
            cm_models = []

            for model in models:
                model.resetting_weight()
                holdout = Holdout(test_size=i/100, trainner=trainner)
                dict_scores = holdout.execute(model, inputs, targets)

                scores_models.append(dict_scores['scores'])
                historys_models.append(dict_scores['history'])
                roc_curve_models.append(dict_scores['roc'])
                cm_models.append(dict_scores['cm'])

            holdout_exp.append((concat_dict(scores_models), historys_models, roc_curve_models, cm_models))
    
    return kfold_exp, holdout_exp

def save_experiment():
    exp_json = Json.dumps(params_exp, indent=4)
    with open(dir_save + 'experiment.json', mode='w') as f:
        f.write(exp_json)

models = initialize_models()
preprocessing_data()
trainner = training()
kfold, holdout = experiment()


print('========================= [Saving models results] ===================================')
print('saving experiment ....')
save_experiment()

for exp_i, exp in enumerate(kfold):
    scores, historys, roc, cms = exp
    save = SaveModel(dir_name=dir_save)

    print('Saving model scores kfold...')
    save.save_results(scores, '{}_kfold10'.format(dataset_name))

    print('Saving models historys kfold...')
    for j, history in enumerate(historys):
        for i, hi in enumerate(history):
            save.save_history_csv(hi,  dataset_name + '_' + models[j]().name + '_k'+ str(i+1))

    print('Saving confusion matrix k-fold')
    for j in range(len(models)):
        path= dir_save + '{}/cm_{}_{}_{}k'
        PlotGraph(
            path=path.format(models[j]().name, dataset_name, models[j]().name, k[exp_i]),
            save=True,
            xlabels=get_classes_names(),
            ylabels=get_classes_names(),
        ).plot_cm(cms[j])
    
    print('Saving roc curve K-fold...')
    for j in range(len(models)):
        fpr, tpr, auc, = roc[j]
        path= dir_save + '{}/roc_{}_{}_{}k'
        PlotGraph(
            path=path.format(models[j]().name, dataset_name, models[j]().name, k[exp_i]),
            save=True,
        ).plot_roc(fpr, tpr, auc)
        # save.save_roc_curve(fpr, tpr, auc, '{}_{}_{}k'.format(dataset_name, models[j]().name, k[exp_i]))
    
    
for i, exp in enumerate(holdout):
    scores, history, roc, cms = exp
    save = SaveModel(dir_name=dir_save)

    print('Saving model scores hold out...')
    save.save_results(scores, 'holdout{}'.format(h[i]))

    print('Saving models historys hold out...')
    for j in range(len(models)):
        save.save_history_csv(history[j], dataset_name + '_' + models[j]().name + '_split'+ str(h[i]))
    
    print('Saving confusion matrix hold out')
    for j in range(len(models)):
        path = dir_save + '{}/cm_{}_{}_{}_split'
        PlotGraph(
            path=path.format(models[j]().name, dataset_name, models[j]().name, h[i]),
            save=True,
            xlabels=get_classes_names(),
            ylabels=get_classes_names(),
        ).plot_cm(cms[j])

    print('Saving roc curve hold out...')
    for j in range(len(models)):
        fpr, tpr, auc, = roc[j]
        path = dir_save + '{}/roc_{}_{}_{}_split'
        PlotGraph(
            path=path.format(models[j]().name, dataset_name, models[j]().name, h[i]),
            save=True
        ).plot_roc(fpr, tpr, auc)

