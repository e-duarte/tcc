import numpy as np
import pandas as pd
from metrics import f1_score
from save_model import SaveModel
from optimizers import Optimizers
from parameters import params_exp
from training_models import Trainner
from tensorflow.keras.datasets import mnist
from validation import KFoldValidation, Holdout, KFoldCustom
from models import Alexnet, Resnet34, DeepAutoencoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from utils import preprocessing, expand_dims, vetorizar_data, to_categorical

models_names = params_exp['models']
optimizer = params_exp['optimizer']
opt_params = params_exp['opt_params']
loss = params_exp['loss']
metrics = params_exp['metrics']
initializers = params_exp['initializers']
epochs = params_exp['epochs']
batch = params_exp['batch_size']
load_model = params_exp['load_model']
data_augmentation = params_exp['data_augmentation']
decay = params_exp['decay_rate']
type = params_exp['type']
dir_save = params_exp['dir_save']
cross = params_exp['cross']
k = params_exp['k-fold']
h = params_exp['holdout']

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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
        model = Resnet34(name=model_name)
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
    
    for name in models_names:
        models.append(build_and_compile_model(name, initializers, (28,28,1), params_compile))
    
    return models

def preprocessing_data():
    global train_images, test_images
    global train_labels, test_labels
    # preprocessing(train_images, test_images)
    if type == 'conv':    
        train_images, test_images = expand_dims(train_images, test_images)
        # train_labels = to_categorical(train_labels)
        # test_labels = to_categorical(test_labels)
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

def apply_validation(models):
    scores_kfold = []
    historys_models = []
    # results_holdout = []
    kfold = KFoldCustom(k=k, trainner=trainner)
    for model in models:
        inputs = np.concatenate((train_images, test_images), axis=0)
        targets = np.concatenate((train_labels, test_labels), axis=0)

        scores_model, history_model = kfold.execute(inputs, targets, model)

        scores_kfold.append(scores_model)
        historys_models.append(history_model)

        # # kfold = KFoldValidation(model,
        # #                         k=10, 
        # #                         train_set=(train_images, train_labels), 
        # #                         test_set=(test_images, test_labels),
        # #                         trainner=trainner)
        # inputs = np.concatenate((train_images, test_images), axis=0)
        # targets = np.concatenate((train_labels, test_labels), axis=0)

        # scores_kfold  = cross_validation(model,inputs, targets, epochs, batch, 10)
        # print(scores_kfold)
        # holdout = Holdout(model,
        #                 train_set=(train_images, test_images), 
        #                 target_set=(train_labels, test_labels),
        #                 trainner=trainner)

        # results_kfold.append(kfold.execute())
        # results_holdout.append(holdout.execute())

    scores_kfold = concat_dict(scores_kfold)
    # results_holdout = concat_dict(results_holdout)

    return scores_kfold, historys_models

models = initialize_models()
preprocessing_data()
trainner = training()

if cross:
    results_kfold, historys_models = apply_validation(models)
    print('\nSaving results for the models =====================================================')
    save = SaveModel(model=None, dir_name=dir_save)
    save.save_results(results_kfold)
    for j, history in enumerate(historys_models):
        for i, h in enumerate(history):
            save.save_history_csv(h, models[j]().name + '_'+ str(i))
    # save.save_results(results_holdout)
else:
    # inputs = np.concatenate((train_images, test_images), axis=0)
    # targets = np.concatenate((train_labels, test_labels), axis=0)

    # kfold = KFoldCustom(10, models[0], trainner)
    # kfold.execute(inputs, targets)

    results = []
    for model in models:
        print('\n------[Training and Evaluate {} model]------------------'.format(model.name))
        trainner.train_model(train_images, train_labels, model())
        print('Avaluating model-------------------------------------------------------------')
        scores = model().evaluate(test_images, test_labels)
        result = {}
        for i, metric in enumerate(model().metrics_names):
                result[metric] = []
                result[metric].append(scores[i])
        result['model'] = [model().name]
        results.append(result)
    results = concat_dict(results)
    print('Saving results for the models')
    save = SaveModel(model=None, dir_name=dir_save)
    save.save_results(results)
    

# model.summary()

# trainner = Trainner(epochs=params_exp['epochs'],
#                     batch_size=params_exp['batch_size'],
#                     data_augmentation=datagen,
#                     callbacks=callbacks)



# 




# if not params_exp['load_weights']:
#     history = trainner.train_model()
# else:
#     history = model.load_weights(params_exp['checkpoint_name'] +'.hdf5')
#     # predicts = model.predict(test_images)
#     # print(np.argmax(predicts[1]), np.argmax(test_labels[1]))
#     # trainner.train_model()
# scores = model.evaluate(test_images, test_labels, verbose=2)
# print(model.metrics_names)
# print(scores)



# history_obj = History(model_name + '_history.csv', history)
# history_obj.save()
# par_dt = pd.DataFrame.from_dict(params_exp)
# with open('parametros_model.json', mode='w') as f:
#     par_dt.to_json(f)

# model(train_images)

'''RESNET'''
# resnet = build_and_compile_model('resnet34')()
# resnet.load_weights('resnet34_model.hdf5')
# resnet.evaluate(test_images,  test_labels, verbose=2)

# resnet.summary()
# train_model(
#     resnet,
#     (train_images, train_labels),
#     checkpoint_name='resnet34_model',
#     validation_data=(test_images, test_labels),
#     batch_size=256)

# resnet.evaluate(test_images,  test_labels, verbose=2)


# resnet.evaluate(train_images,  train_labels, verbose=2)

# train, test = vetorizar_data((train_images, test_images))

# decoder, encoder = build_and_compile_model('autoencoder', optimizer='adam', loss='binary_crossentropy')()

# decoder.load_weights('autoencoder_hinton.hdf5')

# decoder.summary()

# plot_images(encoder, decoder, test)


# train_model(decoder,
#     (train, train),
#     checkpoint_name='autoencoder_hinton',
#     validation_data=(test, test),
#     batch_siz