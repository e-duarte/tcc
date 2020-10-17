import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
import pandas as pd
import seaborn as sn
from datetime import datetime
import os

def normalize(train_images, test_images):
    return train_images.astype('float32')/255, test_images.astype('float32')/255

def expand_dims(train, test):
    return np.expand_dims(train, -1), np.expand_dims(test, -1)

def to_categorical(label_x):
    label_x = utils.to_categorical(label_x, 10)
    # label_y = utils.to_categorical(label_y, 10)

    return label_x

def vetorizar_data(train, test):
    train = train.reshape(len(train), np.prod(train.shape[1:]))
    test = test.reshape(len(test), np.prod(test.shape[1:]))

    return train, test

class PlotGraph:
    def __init__(self, xlabels=None, ylabels=None, path=None, save=False):
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.save = save
        self.path = path

        # dir_path = ''
        # for p in path.split('/')[:-1]:
        #     dir_path += p + '/'
        
        if not os.path.exists(path):            
            print('Making diretory...')
            os.makedirs(path)

        # if save:
        #     self.path =  path + '_' + self.get_date() + '.png'
    
    def get_date(self):
        now = datetime.now()
        return '{}_{}_{}'.format(now.day, now.month, now.year)

    def plot_cm(self, cm, name):
        sn.set()
        cm = pd.DataFrame(cm, index=self.xlabels, columns=self.ylabels)
        _, ax = plt.subplots(figsize=(14, 11))
        graph = sn.heatmap(cm, annot=True, cmap="YlGnBu", fmt='d', linewidths=.9, ax=ax)

        if self.save:
            figure = graph.get_figure()
            figure.savefig(self.path + name, dpi=300)


    def plot_roc(self, fpr, tpr, auc, name):
        # path = self.dir_name + '{}/'.format(name.split('_')[1]) + name + '_' + self.get_date() + '.png'
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.4f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de falso positivo')
        plt.ylabel('Taxa de verdadeiro positivo')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        if self.save:
            plt.savefig(self.path + name, format='png', dpi=300)
        
# def redimensionar(train, test):
#     # train = array_to_img(train[1])
#     # test = array_to_img(test)
#     print(train[1].shape)
#     train = train.resize((256,256), Image.ANTIALIAS)
#     test = test.resize((256,256), Image.ANTIALIAS)
#     train = img_to_array(train)
#     test = img_to_array(test)

#     return train, test


# def plot_images(encoder, autoencoder, data):
#     imagens_codificadas = encoder.predict(data)
#     imagens_decodificadas = autoencoder.predict(data)

#     num_imagens = 10 #numero de imagens exibidas
#     imagens_teste = np.random.randint(data.shape[0], size=num_imagens)
#     plt.figure(figsize=(10,10))
#     for i, indice_imagem in enumerate(imagens_teste):
#         eixo = plt.subplot(10,10, i+1)
#         plt.imshow(data[indice_imagem].reshape(28,28))
#         plt.xticks(())
#         plt.yticks(())

#         eixo = plt.subplot(10,10, i+1+num_imagens)
#         plt.imshow(imagens_codificadas[indice_imagem].reshape(6,5))
#         plt.xticks(())
#         plt.yticks(())

#         eixo = plt.subplot(10,10, i+1+num_imagens*2)
#         plt.imshow(imagens_decodificadas[indice_imagem].reshape(28,28))
#         plt.xticks(())
#         plt.yticks(())

#     plt.show()



        
