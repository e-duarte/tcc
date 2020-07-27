from tensorflow.keras import datasets
from utils import vetorizar_data, expand_dims, normalize
import numpy as np


class DatasetFactory:
    def __init__(self, name, flat=False, concat=False, expand=False, normalize=False):
        if 'mnist' == name:
            (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        elif 'cifar10' == name:
            (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        
        if normalize:
            x_train, x_test = normalize(x_train, x_test)

        if flat:
            x_train, x_test = vetorizar_data(x_train, x_test)
        
        if expand:
            print('sim')
            expand_dims(x_train, x_test)

        if concat:
            inputs = np.concatenate((x_train, x_test), axis=0)
            targets = np.concatenate((y_train, y_test), axis=0)

            self.dataset = (inputs, targets)
        else:
            self.dataset = ((x_train, y_train), (x_test, y_test))

        self.shape = self.get_shape(name, x_train)
        self.classes_names = self.get_classes_names(name)


    def get_classes_names(self, name):
        return [i for i in range(10)] if name == 'mnist' else ['airplane', 'automobile',
                                                                            'bird', 'cat', 'deer',
                                                                            'dog', 'frog', 'horse', 
                                                                            'ship', 'truck']
    
    def get_shape(self, name, x):
        return x.shape[1:] if 'mnist' != name else x.shape[1:] + (1,) 
