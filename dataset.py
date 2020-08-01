from tensorflow.keras import datasets
from utils import vetorizar_data, expand_dims, normalize
import numpy as np
from sklearn.utils import shuffle


class DatasetFactory:
    def __init__(self, name, flat=False, concat=False, expand=False, normalize=False):
        self.name = name
        
        (x_train, y_train), (x_test, y_test) = self.load_data(name)
        
        if normalize:
            x_train, x_test = normalize(x_train, x_test)

        if flat:
            x_train, x_test = vetorizar_data(x_train, x_test)
        
        if expand:
            expand_dims(x_train, x_test)

        if concat:
            inputs, targets = self.concat(x_train, y_train, x_test, y_test)

            self.dataset = (inputs, targets, [i for i in range(len(inputs))])
        else:
            self.dataset = ((x_train, y_train), (x_test, y_test), len(x_train))

        self.shape = self.get_shape(name, x_train)
        self.classes_names = self.get_classes_names(name)

    def load_data(self, name):
        if 'mnist' == name:
            return datasets.mnist.load_data()
        elif 'cifar10' == name:
            return datasets.cifar10.load_data()
    
    def concat(self, x_train, y_train, x_test, y_test):
        inputs = np.concatenate((x_train, x_test), axis=0)
        targets = np.concatenate((y_train, y_test), axis=0)

        return (inputs, targets)

    def get_classes_names(self, name):
        return [i for i in range(10)] if name == 'mnist' else ['airplane', 'automobile',
                                                                            'bird', 'cat', 'deer',
                                                                            'dog', 'frog', 'horse', 
                                                                            'ship', 'truck']
    
    def get_shape(self, name, x):
        return x.shape[1:] if 'mnist' != name else x.shape[1:] + (1,) 
    
    def shuffle(self):
        (inputs, targets, indexs) = self.dataset
        inputs, targets, indexs = shuffle(inputs, targets, indexs)
        
        self.dataset = (inputs, targets, indexs)
    
    def restore_state(self, indexs):
        (x_train, y_train), (x_test, y_test) = self.load_data(self.name)
        inputs, targets = self.concat(x_train, y_train, x_test, y_test)

        inputs = np.array(list(map(lambda i: inputs[i], indexs)))
        targets =  np.array(list(map(lambda i: targets[i], indexs)))

        self.dataset = (inputs, targets, indexs)
