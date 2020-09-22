from experiment import Experiment
from dataset import DatasetFactory
import json
import sys

if __name__ == '__main__':
    with open(sys.argv[1], mode='r') as f:
        experiment_params = json.loads(f.read())
    
    experiment = Experiment(experiment_params)
    experiment.execute()