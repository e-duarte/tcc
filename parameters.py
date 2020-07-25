import sys

params_exp = {
    'models': ['alexnet'],
    'dataset': sys.argv[1],
    'optimizer': 'sgd',
    'opt_params': {'learning_rate':0.01, 'momentum':0.9, 'decay':0.1e-3},
    # 'opt_params': {},
    'loss': 'categorical_crossentropy',
    'metrics': ['categorical_accuracy',
                'Precision',
                'Recall',
                'TruePositives', 
                'FalsePositives', 
                'TrueNegatives', 
                'FalseNegatives',
                'AUC',
                'f1_score'],
    'initializers': False,
    'epochs': 1,
    'batch_size': 256,
    'load_model': False,
    'data_augmentation':False,
    'decay_rate': True,
    'type':'conv',
    'dir_save': '/content/drive/My Drive/TCC/results/experiment_convs/',
    # 'cross': True,
    'k-fold': [2],
    'holdout': [],
}

# print(params_exp)
# print(type(params_exp))
# exit()
