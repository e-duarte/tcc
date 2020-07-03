import tensorflow as tf
from tensorflow import keras
from tensorflow import data
from models import Alexnet
import numpy as np

train, test = keras.datasets.mnist.load_data()

images, labels = train
images = images/255
labels = labels.astype(np.int32)
images = np.expand_dims(images, -1)

dataset = data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(60000).batch(128)

alexnet = Alexnet()
model = alexnet()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')
model.fit(dataset, epochs=10)