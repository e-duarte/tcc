params_exp = {
    'model_name':'alexnet',
    'params_compile': {
        'optimizer':{'name':'sgd',
                    'params':{'learning_rate':0.01,
                                'momentum':0.9,
                                'decay':0.5e-3}},
        'loss':'categorical_crossentropy',
        'metrics':['accuracy'],
                    # 'Precision',
                    # 'Recall',
                    # 'TruePositives',
                    # 'FalsePositives',
                    # 'TrueNegatives',
                    # 'FalseNegatives']
    },
    'initializers':True,
    'data_augmentation':True,
    'decay_rate':True,
    'checkpoint_name':'alexnet_save_28x28',
    'tensor_board':False,
    'epochs':90,
    'batch_size': 128,
    'load_weights': False,
    'resize':(28,28,1)
}

params_exp = {
    'model_name':'resnet',
    'params_compile': {
        'optimizer':{'name':'sgd',
                    'params':{'learning_rate':0.1,
                                'momentum':0.9,
                                'decay':0.1e-3}},
        'loss':'categorical_crossentropy',
        'metrics':['accuracy'],
                    # 'Precision',
                    # 'Recall',
                    # 'TruePositives',
                    # 'FalsePositives',
                    # 'TrueNegatives',
                    # 'FalseNegatives']
    },
    'initializers':False,
    'data_augmentation':True,
    'decay_rate':True,
    'checkpoint_name':'resnet_save_28x28',
    'tensor_board':False,
    'epochs':90,
    'batch_size': 256,
    'load_weights': False,
    'resize':(28,28,1)
}