import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D,  Dense, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D,  AveragePooling2D


class BaseModel:
    def __init__(self):
        self.model = None

    def build(self):
        print("__build__")
    
    def compile(self, params):
        self.model.compile(
            optimizer=params['optimizer'],
            loss=params['loss'],
            metrics=params['metrics']
        )

    def __call__(self):
        return self.model

class Alexnet(BaseModel):
    def __init__(self, input_shape=(28,28,1), initializers=True):
        # self.model = None
        normal = None
        one = None
        zero = None
        kernel_pooling = (3,3)
        stride_pooling = (2,2)
        pad = 'same'
        activation = 'relu'
        one = None
        zero = None
        
        if initializers:
            normal = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)
            one = keras.initializers.Ones()
            zero = keras.initializers.Zeros()
            # one = None

        

        self.input_image = Input(shape=input_shape,
                            name="input_images")
        #layers
        self.conv_1 = Conv2D(96, (5,5), name='1Conv',
                            activation = activation,
                            kernel_initializer=normal,
                            bias_initializer=zero,
                            strides=(2,2),
                            padding=pad)
        
        self.max_pooling_1 = MaxPooling2D(pool_size=kernel_pooling,
                                        name='1maxpooling',
                                        strides=stride_pooling,
                                        padding=pad)

        self.conv_2 = Conv2D(256, (5,5), name='2Conv',
                            activation = activation,
                            kernel_initializer=normal,
                            bias_initializer=one,
                            strides=(4,4),
                            padding=pad)

        self.max_pooling_2 = MaxPooling2D(pool_size=kernel_pooling,
                                        name='2maxpooling',
                                        strides=stride_pooling,
                                        padding=pad)

        self.conv_3 = Conv2D(384, (3,3), name='3Conv',
                            activation = activation,
                            kernel_initializer=normal,
                            bias_initializer=zero,
                            strides=(4,4),
                            padding=pad)

        self.conv_4 = Conv2D(384, (3,3), name='4Conv',
                            activation = activation,
                            kernel_initializer=normal,
                            bias_initializer=one,
                            strides=(4,4),
                            padding=pad)

        self.conv_5 = Conv2D(256, (3,3), name='5Conv',
                            activation = activation,
                            kernel_initializer=normal,
                            bias_initializer=one,
                            strides=(4,4),
                            padding=pad)

        self.max_pooling_3 = MaxPooling2D(pool_size=kernel_pooling,
                                        name='3maxpooling',
                                        strides=stride_pooling,
                                        padding=pad)

        self.dense_1 = Dense(4096, activation = activation,
                            name='1dense',
                            kernel_initializer=normal,
                            bias_initializer=one)

        self.dense_2 = Dense(4096, activation = activation,
                            name='2dense',
                            kernel_initializer=normal,
                            bias_initializer=one)

        self.dense_3 = Dense(10, activation = 'softmax',
                            name='classifier',
                            kernel_initializer=normal,
                            bias_initializer=zero)

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
        self.model = Model(inputs=self.input_image, outputs=output)
        # return self.model

    # def __call__(self):
    #     return self.model

class Resnet34(BaseModel):
    def __init__(self, input_shape=(28,28,1)):
        self.num = 64
        self.blocks = [3, 4, 6, 3] #Define a arquitetura da rede
        self.count = 1
        self.label = 'Conv_{}'
        self.input_shape = input_shape
        
        # self.model = None

        self.build()
        
    def build(self):
        input = Input(shape=self.input_shape)

        x = Conv2D(64,
                  (7,7),
                  strides=2,
                  padding='same',
                  name= self.label.format(self.count),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((3,3),
                        strides=(2,2), 
                        padding='same',
                        name='MaxPooling')(x)

        downsample = False
        self.count += 1
        for stage in self.blocks:
            if downsample:
                stride = 2 if stage == 6 else 1
                x = self.downsample_block(self.num, x, stride)
                stage -= 1
                downsample = False
            x = self.identity_block(stage, self.num, x)
            self.num *= 2  
            downsample = True

        x = AveragePooling2D(pool_size=2, name='AveragePooling')(x)
        x = Flatten()(x)
        x = Dense(10,
                 activation='softmax',
                 kernel_initializer='he_normal',
                 name='classifier')(x)
        self.model = Model(inputs=input, outputs=x)
    
    def identity_block(self, numblocks, num_filters, x):
        for i in range(numblocks):
            y = Conv2D(num_filters,
                      (3,3),
                      padding='same',
                      name=self.label.format(self.count),
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            self.count +=1

            y = Conv2D(num_filters,
                      (3,3),
                      padding='same',
                      activation='relu',
                      name=self.label.format(self.count),
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(y)
            x = BatchNormalization()(x)

            self.count +=1

            z = keras.layers.add([x,y])
            x = Activation('relu')(z)
        return x

    def downsample_block(self, num_filters, x, stride):
        y = Conv2D(num_filters,
                  (3,3),
                  padding='same',
                  activation='relu',
                  strides=stride,
                  name=self.label.format(self.count),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        self.count +=1

        y = Conv2D(num_filters,
                  (3,3),
                  padding='same',
                  activation='relu',
                  name=self.label.format(self.count),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
        x = BatchNormalization()(x)

        self.count +=1

        down = Conv2D(num_filters,
                     (1,1),
                     padding='same',
                     strides=stride,
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))(x)

        z = keras.layers.add([down,y])
        x = Activation('relu')(z)
        return x

    # def __call__(self):
    #     return self.model

class DeepAutoencoder(BaseModel):
    def __init__(self, input_shape=(784,)):
        self.hidden_activation = 'relu'
        self.output_activation = 'sigmoid'
        self.architecture_encoder = [1000, 500, 250, 30]
        self.architecture_decoder = [250, 500, 1000, 784]
        self.input_shape = input_shape

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

        self.model = Model(inputs=input, outputs=y, name='Autoencoder')
    
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

# (autoencoder, encoder) = DeepAutoencoder()()
# autoencoder.summary()
# encoder.summary()

