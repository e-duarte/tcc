import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

def preprocessing(train_images, test_images):
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