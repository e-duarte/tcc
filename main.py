from experiment import Experiment
from dataset import DatasetFactory
import json
import sys

if __name__ == '__main__':
    with open(sys.argv[1], mode='r') as f:
        experiment_params = json.loads(f.read())
    
    expand = False
    flat = False
    if experiment_params['type'] == 'conv':
        expand = True
    else:
        flat = False
    dataset = DatasetFactory(
        experiment_params['dataset'],
        flat=flat,
        concat=True,
        expand=expand
    )

    experiment = Experiment(experiment_params, dataset)
    experiment.execute()