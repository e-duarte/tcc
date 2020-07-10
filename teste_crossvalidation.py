from sklearn.model_selection import cross_val_score,  ShuffleSplit, cross_validate, KFold
from models import Alexnet
from tensorflow.keras.datasets import mnist
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from utils import expand_dims, preprocessing, to_categorical
from sklearn.metrics import recall_score, precision_score, accuracy_score
import numpy as np
from validation import KFoldCustom

# scoring = ['precision_macro', 'recall_macro']
# # scoring = {
# #             'accuracy':accuracy_score,
# #             'precision':precision_score,
# #             'recall':recall_score}

(train_x, train_y), (test_x, test_y) = mnist.load_data()

print(train_y.shape)



train_x, test_x = expand_dims(train_x, test_x)
# train_x, test_x = preprocessing(train_x, test_y)
# print(train_x.shape)
train_y, test_y = to_categorical(train_y), to_categorical(test_y)

# # inputs = np.concatenate((train_x, test_x), axis=0)
# # targets = np.concatenate((train_y, test_y),  axis=0)

# # train[0] = train[0]/255

def build_model():
    model = Alexnet(initializers=False)()

    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# # cv =  ShuffleSplit(n_splits=5, random_state=0)
# # model = KerasClassifier(build_model, epochs=1, batch_size=128)

# # scores = cross_val_score(model,  inputs, targets, cv=cv, scoring=accuracy_score)
# # scores = cross_validate(model,  inputs, targets, cv=cv, scoring=scoring)

# # print(scores['test_accuracy_score'].mean())
# # print(scores)



# def kfold(model):
#     cross = KFold(n_splits=5, shuffle=True)
    
#     for train, test in cross.split(test_x, test_y):
#         model.fit(
#             test_x[train],
#             test_y[train],
#             epochs=1, 
#             batch_size=128, 
#             # validation_data=(inputs[test], targets[test])
#         )

# kfold(build_model())

# def custom_cv_folds(X, cv=2):
#     n = X.shape[0]
#     i = 1
#     dataset = np.arange(0, n, dtype=int)
#     while i <= cv:
#         idx = np.arange(n * (i - 1) / cv, n * i / cv, dtype=int)
#         yield np.array(list(set(dataset) - set(idx))), idx
#         i += 1


# for obj in custom_cv_folds(train_x, cv=2):
#     print(obj)

model = build_model()
kfold = KFoldCustom(10, model)
# kfold.execute(train_x, train_y)
for train, test in kfold.split(train_x):
    train, test = train_x[train], train_x[test]
    target_train, target_test = train_y[train], train_y[test]

    model.fit(train, target_train, epochs=1, batch_size=128)