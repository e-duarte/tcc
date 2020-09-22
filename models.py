import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Conv2D,  Dense, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D,  AveragePooling2D


class BaseModel:
    def __init__(self):
        self.model = None
        self.weight_initial = None

    def build(self):
        print("__build__")
    
    def compile(self, params):
        self.model.compile(
            optimizer=params['optimizer'],
            loss=params['loss'],
            metrics=params['metrics']
        )
        self.weight_initial = self.model.get_weights()
    
    def resetting_weight(self):
        self.model.set_weights(self.weight_initial)

    def __call__(self):
        return self.model

class FactoryModel:
    def __init__(self, name, nick, size, params_compile, initializers=False):
        self.model = None
        if name == 'alexnet':
            self.model = Alexnet(input_shape=size, initializers=initializers, name=nick)
        elif name == 'resnet':
            self.model = Resnet34(input_shape=size, name=nick)
        elif name == 'autoencoder':
            self.model = DeepAutoencoder(name=nick)
        elif name == 'test':
            self.model = ModelTest(name=nick)
            
        # self.model().summary()
        self.model.compile(params_compile)
        
    def get_model(self):
        return self.model

class Alexnet(BaseModel):
    def __init__(self, input_shape=(28,28,1), initializers=True, name='model'):
        initializer = None
        ones  = None
        zeros = None
        kernel_pooling = (3,3)
        stride_pooling = (2,2)
        pad = 'same'
        activation = 'relu'
        self.name = name
        
        if initializers:
            initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)
            ones  = keras.initializers.Ones()
            zeros = keras.initializers.Zeros()

        

        self.input_image = Input(shape=input_shape,
                            name="input_images")
        #layers
        self.conv_1 = Conv2D(96, (11,11), name='1Conv',
                            activation = activation,
                            kernel_initializer=initializer,
                            bias_initializer=zeros,
                            strides=(4,4),
                            padding=pad)
        
        self.max_pooling_1 = MaxPooling2D(pool_size=kernel_pooling,
                                        name='1maxpooling',
                                        strides=stride_pooling,
                                        padding=pad)

        self.conv_2 = Conv2D(256, (5,5), name='2Conv',
                            activation = activation,
                            kernel_initializer=initializer,
                            bias_initializer=ones,
                            # strides=(4,4),
                            padding=pad)

        self.max_pooling_2 = MaxPooling2D(pool_size=kernel_pooling,
                                        name='2maxpooling',
                                        strides=stride_pooling,
                                        padding=pad)

        self.conv_3 = Conv2D(384, (3,3), name='3Conv',
                            activation = activation,
                            kernel_initializer=initializer,
                            bias_initializer=zeros,
                            # strides=(4,4),
                            padding=pad)

        self.conv_4 = Conv2D(384, (3,3), name='4Conv',
                            activation = activation,
                            kernel_initializer=initializer,
                            bias_initializer=ones,
                            # strides=(4,4),
                            padding=pad)

        self.conv_5 = Conv2D(256, (3,3), name='5Conv',
                            activation = activation,
                            kernel_initializer=initializer,
                            bias_initializer=ones,
                            # strides=(4,4),
                            padding=pad)

        self.max_pooling_3 = MaxPooling2D(pool_size=kernel_pooling,
                                        name='3maxpooling',
                                        strides=stride_pooling,
                                        padding=pad)

        self.dense_1 = Dense(4096, activation = activation,
                            name='1dense',
                            kernel_initializer=initializer,
                            bias_initializer=ones )

        self.dense_2 = Dense(4096, activation = activation,
                            name='2dense',
                            kernel_initializer=initializer,
                            bias_initializer=ones )

        self.dense_3 = Dense(10, activation = 'softmax',
                            name='classifier',
                            kernel_initializer=initializer,
                            bias_initializer=ones)

        self.build()

    def build(self):
        x = self.conv_1(self.input_image)
        x = tf.nn.lrn(x,
                    alpha=1e-4,
                    beta=0.75,
                    depth_radius=2,
                    bias=2.0)
        x = self.max_pooling_1(x)
        x = self.conv_2(x)
        x = tf.nn.lrn(x,
                    alpha=1e-4,
                    beta=0.75,
                    depth_radius=2,
                    bias=2.0)
        x = self.max_pooling_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.max_pooling_3(x)
        x = Flatten()(x)
        x = self.dense_1(x)
        x = Dropout(0.5, name='1Dropout')(x)
        x = self.dense_2(x)
        x = Dropout(0.5, name='2Dropout')(x)
        output = self.dense_3(x)
        self.model = Model(inputs=self.input_image, outputs=output, name=self.name)
        # return self.model

class Resnet34(BaseModel):
    def __init__(self, input_shape=(28,28,1), name='model'):
        self.num = 64
        self.blocks = [3, 4, 6, 3] #Define a arquitetura da rede
        self.count = 1
        self.label = 'Conv_{}'
        self.input_shape = input_shape
        self.name = name
        # self.model = None

        self.build()
        
    def build(self):
        input = Input(shape=self.input_shape)

        name = self.label.format(self.count)
        x = Conv2D(64,
                  (7,7),
                  strides=2,
                  padding='same',
                  name=name,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(input)

        name = "BN_{}".format(self.count)                  
        x = BatchNormalization(name=name)(x)

        name = "RELU_{}".format(self.count)
        x = Activation('relu', name=name)(x)

        name = "MXP_{}".format(self.count)
        x = MaxPooling2D((3,3),
                        strides=(2,2), 
                        padding='same',
                        name=name)(x)

        downsample = False
        self.count += 1
        for stage in self.blocks:
            if downsample:
                stride = 2
                x = self.downsample_block(self.num, x, stride)
                stage -= 1
                downsample = False
            # else:
            x = self.identity_block(stage, self.num, x)
            self.num *= 2  
            downsample = True

        x = AveragePooling2D(pool_size=1, name='AVG')(x)
        x = Flatten()(x)
        x = Dense(10,
                 activation='softmax',
                 kernel_initializer='he_normal',
                 name='classifier')(x)
        self.model = Model(inputs=input, outputs=x, name=self.name)
    
    def identity_block(self, numblocks, num_filters, x):
        for i in range(numblocks):
            name = self.label.format(self.count)
            y = Conv2D(num_filters,
                      (3,3),
                      padding='same',
                      name=name,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(x)
            
            name = "BN_{}".format(self.count)                  
            y = BatchNormalization(name=name)(y)

            name = "RELU_{}".format(self.count)
            y = Activation('relu', name=name)(y)

            self.count +=1

            name = self.label.format(self.count)
            y = Conv2D(num_filters,
                      (3,3),
                      padding='same',
                      activation='relu',
                      name=name,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(y)
                      
            name = "BN_{}".format(self.count)                  
            y = BatchNormalization(name=name)(y)

            z = keras.layers.add([x,y])

            name = "RELU_{}".format(self.count)
            x = Activation('relu', name=name)(z)
            self.count +=1
        return x

    def downsample_block(self, num_filters, x, stride):
        name = self.label.format(self.count)
        y = Conv2D(num_filters,
                  (3,3),
                  padding='same',
                  activation='relu',
                  strides=stride,
                  name=name,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

        name = "BN_{}".format(self.count)                  
        y = BatchNormalization(name=name)(y)

        name = "RELU_{}".format(self.count)
        y = Activation('relu', name=name)(y)

        self.count +=1
        y = Conv2D(num_filters,
                  (3,3),
                  padding='same',
                  activation='relu',
                  name=self.label.format(self.count),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
        
        name = "BN_{}".format(self.count)                  
        y = BatchNormalization(name=name)(y)

        down = Conv2D(num_filters,
                     (1,1),
                     padding='same',
                     name='Down_{}'.format(self.count),
                     strides=stride,
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))(x)

        z = keras.layers.add([down,y])
        name = "RELU_{}".format(self.count)
        x = Activation('relu', name=name)(z)
        
        self.count +=1
        return x

class DeepAutoencoder(BaseModel):
    def __init__(self, input_shape=(784,), name='model'):
        self.hidden_activation = 'relu'
        self.output_activation = 'sigmoid'
        self.architecture_encoder = [1000, 500, 250, 30]
        self.architecture_decoder = [250, 500, 1000, 784]
        self.input_shape = input_shape
        self.name = name

        self.build()

    def build(self):
        input  = Input(shape=self.input_shape, name='Input_Layer')

        label = 'Dense_{}'

        x = Dense(self.architecture_encoder[0],
                  activation=self.hidden_activation,
                  name=label.format(0))(input)

        
        count = 1
        for units in self.architecture_encoder[1:]:
            x = Dense(units,
                      activation=self.hidden_activation,
                      name=label.format(count))(x)
            count += 1
        
        for units in self.architecture_decoder[:3]:
            x = Dense(units,
                      activation=self.hidden_activation,
                      name=label.format(count))(x)
            count += 1

        y = Dense(self.architecture_decoder[-1],
                 activation=self.output_activation,
                 name='Output_Layer')(x)

        self.model = Model(inputs=input, outputs=y, name=self.name)
    
    def encoder(self):
        input = Input(shape=self.input_shape, name='Input_Layer')

        layer_0 = self.model.get_layer(name='Dense_0')
        x = layer_0(input)

        layer_1 = self.model.get_layer(name='Dense_1')
        x = layer_1(x)

        layer_2 = self.model.get_layer(name='Dense_2')
        x = layer_2(x)
    
        layer_3 = self.model.get_layer(name='Dense_3')
        y = layer_3(x)

        return Model(inputs=input, outputs=y, name='Encoder')
    
    def __call__(self):
        return (self.model, self.encoder())

class ModelTest(BaseModel):
    def __init__(self, name):
        self.model = Sequential([
            Flatten(input_shape=(28,28,1)),
            Dense(10, activation='softmax', bias_initializer='ones')
        ], name=name)

        


if __name__ == '__main__':
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.callbacks import CSVLogger
    import json
    import pandas as pd
    import gzip, os

    (x, y), (_, _) = mnist.load_data()

    model = ModelTest('testeModel')()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x, y, epochs=10, callbacks=[CSVLogger('history.csv')])
    os.system('scp -P 16280 ./history.csv ewerton@0.tcp.ngrok.io:/home/ewerton')