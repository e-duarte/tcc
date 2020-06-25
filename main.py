import numpy as np
import pandas as pd
from save_model import SaveModel
from optimizers import Optimizers
from parameters import params_exp
from training_models import Trainner
from tensorflow.keras.datasets import mnist
from crossvalidation import KFoldValidation
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

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


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
    preprocessing(train_images, test_images)
    if type == 'conv':    
        train_images, test_images = expand_dims(train_images, test_images)
        train_labels, test_labels = to_categorical(train_labels, test_labels)
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
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, min_lr=0.1e-4))
                                    
    return Trainner(epochs=epochs,batch_size=batch, data_augmentation=datagen, callbacks=callbacks)

models = initialize_models()
preprocessing_data()
trainner = training()


for model in models:
    kfold = KFoldValidation(model, 
                            train_set=(train_images, test_images), 
                            target_set=(train_labels, test_labels),
                            trainner=trainner)
    kfold.execute()
    print('Saving the {}'.format(model.name))
    SaveModel(model(), dir_name=dir_save)



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