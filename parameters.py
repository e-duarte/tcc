params_exp = {
    'models': ['alexnet', 'resnet'],
    'optimizer': 'sgd',
    'opt_params': {'learning_rate':0.01, 'momentum':0.9, 'decay':0.5e-3},
    'loss': 'categorical_crossentropy',
    'metrics': ['categorical_accuracy', 'Precision'],
    'initializers': True,
    'epochs': 1,
    'batch_size': 256,
    'load_model': False,
    'data_augmentation':False,
    'decay_rate': True,
    'type':'conv'
}
