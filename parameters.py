params_exp = {
    'model_name':'alexnet',
    'params_compile': {
        'optimizer':'adam',
        'loss':'categorical_crossentropy',
        'metrics':['accuracy', 'Precision']
    },
    'optimizer':{'learning_rate':0.01, 'momentum':0.9},
    'initializers':True,
    'data_augmentation':True,
    'decay_rate':True,
    'checkpoint_name':'alexnet_save_modify',
    'tensor_board':False,
    'epochs':90,
    'batch_size': 128,
    'load_weights': True,
    'resize':(28,28,1)
}
