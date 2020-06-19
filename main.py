from models import Alexnet, Resnet34, DeepAutoencoder
from utils import preprocessing, expand_dims, vetorizar_data, to_categorical, redimensionar
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from parameters import params_exp
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from history import History
from training_models import Trainner
from tensorflow.image import resize
import numpy as np
import pandas as pd
from optimizers import Optimizers


def build_and_compile_model(model_name, initializers, size, params_compile={
                    'optimizer':'adam',
                    'loss':'categorical_crossentropy',
                    'metrics':['accuracy']}):
    model = None
    print('[BUILDING MODEL] {} model'.format( model_name))
    if model_name == 'alexnet':
        model = Alexnet(input_shape=size, initializers=initializers)
    elif model_name == 'resnet':
        model = Resnet34()
    elif model_name == 'autoencoder':
        model = DeepAutoencoder()
    print('[SUCCESSFUL] {} build'.format( model_name))

    model.compile(params_compile)
    print('[COMPILED] {} compiled'.format( model_name))

    return model

def scheduler_decay(epochs, lr):
    print('decay')
    return lr*5e-4


model_name = params_exp['model_name']

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
preprocessing(train_images, test_images)

datagen = None
callbacks = []
size = params_exp['resize']
history = None
params_compile = params_exp['params_compile']
params_compile['optimizer'] = Optimizers(params_compile['optimizer']['name'], 
                                        params_compile['optimizer']['params']).optimizer()



if model_name == 'alexnet' or model_name == 'resnet':    
    train_images, test_images = expand_dims(train_images, test_images)
    train_labels, test_labels = to_categorical(train_labels, test_labels)
    train_images, test_images = resize(train_images, size[:2]), resize(test_images, size[:2])
else:
    vetorizar_data(train_images, test_images)

if model_name == 'alexnet':
    if params_exp['data_augmentation']:
        datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True)
    if params_exp['decay_rate']:
        scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, min_lr=0.1e-4)
        callbacks.append(scheduler)
        # callbacks.append(LearningRateScheduler(scheduler_decay))
    
model = build_and_compile_model(model_name,params_exp['initializers'], size, params_compile=params_compile)()
model.summary()

trainner = Trainner(
                model = model,
                dataset=(train_images, train_labels),
                checkpoint_name=params_exp['checkpoint_name'],
                epochs=params_exp['epochs'],
                validation_data=(test_images, test_labels),
                batch_size=params_exp['batch_size'],
                data_augmentation=datagen,
                callbacks=callbacks,
                tensor_board=params_exp['tensor_board'])

if not params_exp['load_weights']:
    history = trainner.train_model()
else:
    history = model.load_weights(params_exp['checkpoint_name'] +'.hdf5')
    # predicts = model.predict(test_images)
    # print(np.argmax(predicts[1]), np.argmax(test_labels[1]))
    # trainner.train_model()

history_obj = History(model_name + '_history.csv', history)
history_obj.save()
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
#     batch_size=256,
#     epochs=50
# )