from sklearn.model_selection import cross_val_score,  ShuffleSplit, cross_validate
from models import Alexnet
from tensorflow.keras.datasets import mnist
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from utils import expand_dims, preprocessing, to_categorical
from sklearn.metrics import recall_score, precision_score, accuracy_score
import numpy as np

# scoring = [accuracy_score, precision_score, recall_score]
scoring = {
            'accuracy':accuracy_score,
            'precision':precision_score,
            'recall':recall_score}

(train_x, train_y), (test_x, test_y) = mnist.load_data()



train_x, test_x = expand_dims(train_x, test_x)
# train_x, test_x = preprocessing(train_x, test_y)
train_y, test_y = to_categorical(train_y), to_categorical(test_y)

inputs = np.concatenate((train_x, test_x), axis=0)
targets = np.concatenate((train_y, test_y),  axis=0)

# train[0] = train[0]/255

def build_model():
    model = Alexnet(initializers=False)()

    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision']
    )

    return model


cv =  ShuffleSplit(n_splits=5, random_state=0)
model = KerasClassifier(build_model, epochs=2, batch_size=256)

# scores = cross_val_score(model,  inputs, targets, cv=cv)
scores = cross_validate(model,  inputs, targets, cv=cv, scoring=scoring)

print(scores['test_accuracy_score'].mean())
print(scores)