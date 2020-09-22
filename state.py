import os, json

class LoaderState:
    def __init__(self, id_exp, epochs, dataset, valid_exp, url):
        self.state = None
                
        path = url + 'states/state_{}.json'.format(id_exp)
        if os.path.exists(path):
            from_dict_state = self.load_json(path)
            self.state = ExperimentState(from_dict=from_dict_state)
            dataset.restore_state(self.state.dataset)
        else:
            dataset.shuffle()
            self.state = ExperimentState(id_exp=id_exp,
                                        epochs_max=epochs,
                                        dataset=dataset.dataset[2],
                                        valid_exp=valid_exp,
                                        url=url,
                                        )
    def load_json(self, url):
        state = {}
        with open(url, mode='r') as f:
            state = json.loads(f.read())
        return state

class ExperimentState:
    def __init__(self,
                id_exp=None,
                epochs_max=None,
                dataset=None,
                valid_exp=None,
                url='',
                from_dict= {}):

        if from_dict != {}:
            print('[INFO] LOADING STATE...')

            if os.path.exists('./tmp_history.csv'):
                os.remove('./tmp_history.csv')
            

            self.status = from_dict['status']
            self.id_exp = from_dict['id_exp']
            self.epochs_max = from_dict['epochs_max']

            valids = {k:[] for k in from_dict['valid_exp']}
            for v in valids:
                if v == 'holdout':
                    for h in from_dict['valid_exp'][v]:
                        valids[v].append(HoldOutState(from_dict=h))
                elif v == 'kfold':
                    for k in from_dict['valid_exp'][v]:
                        valids[v].append(KFoldState(from_dict=k))

            self.valid_exp = valids
            self.dataset = from_dict['dataset']
        else:
            print('[INFO] MAKE NEW STATE...')

            if os.path.exists('./tmp_history.csv'):
                os.remove('./tmp_history.csv')

            valids = {k:[] for k in valid_exp}
            for v in valid_exp:
                if v == 'holdout':
                    for h in valid_exp[v]:
                        valids[v].append(HoldOutState(split=h, url=url))
                elif v == 'kfold':
                    for k in valid_exp[v]:
                        valids[v].append(KFoldState(k=k, url=url))

            self.status = False
            self.id_exp = id_exp
            self.epochs_max = epochs_max
            self.valid_exp = valids
            self.dataset = dataset
    
    def get_validation(self, valid_name):
        validation = self.valid_exp[valid_name]
        
        valid_exps = []

        if valid_name == 'holdout':
            for state_h in validation:
                if not state_h.status:
                    valid_exps.append(state_h.split/100)
        else:
            for state_k in validation:
                if not state_k.status:
                    valid_exps.append(state_k.k)
        
        return valid_exps

    def get_state_validation(self, valid_name, k=None, h=None):
        validation = None
        
        if k != None:
            for v in self.valid_exp[valid_name]:
                if v.k == k:
                    validation = v
        elif h != None:
            for v in self.valid_exp[valid_name]:
                if v.split == h:
                    validation = v

        return validation
    
    def to_dict(self):
        valids = {k:[] for k in self.valid_exp}
        
        for v in self.valid_exp:
            for i in self.valid_exp[v]:
                valids[v].append(i.to_dict())
        
        state = {
            'status': self.status,
            'id_exp': self.id_exp,
            'epochs_max': self.epochs_max,
            'valid_exp': valids,
            'dataset': self.dataset,
        }
        return state

    def __str__(self):
        return str(self.to_dict())
        
class HoldOutState:
    def __init__(self, split=10, url='', from_dict={}):
        if from_dict == {}:
            self.epochs = 0
            self.split = split
            self.weights = url + 'weights/'
            self.history = url + 'results/'
            self.status = False
        else:
            self.epochs = from_dict['epochs']
            self.split = from_dict['split']
            self.weights = from_dict['weights']
            self.history = from_dict['history']
            self.status = from_dict['status']
    
    def to_dict(self):
        return {
            'status': self.status,
            'epochs': self.epochs,
            'split': self.split,
            'weights': self.weights,
            'history': self.history,
        }
    
    def __str__(self):
        return 'holdout'

class KFoldState:
    def __init__(self, k=10, url='', from_dict={}):
        if from_dict == {}:
            self.epochs = [0 for i in range(k)]
            self.k = k
            self.current_k = 1
            self.weights = [url + 'weights/' for i in range(k)]
            self.historys = [url + 'results/' for i in range(k)]
            self.status = False
        else:
            self.epochs = from_dict['epochs']
            self.k = from_dict['k']
            self.current_k = from_dict['current_k']
            self.weights = from_dict['weights']
            self.historys = from_dict['historys']
            self.status = from_dict['status']
    
    def get_epochs(self, i):
        return self.epochs[i]

    def get_weights(self, i):
        return self.weights[i]
    
    def increment_current_k(self):
        self.current_k += 1

    def get_history(self, i):
        return self.historys[i]
    
    def to_dict(self):
        return {
            'status': self.status,
            'epochs': self.epochs,
            'k': self.k,
            'current_k': self.current_k,
            'weights': self.weights,
            'historys': self.historys,
        }
    
    def __str__(self):
        return 'kfold'

if __name__ == '__main__':
    from dataset import DatasetFactory
    from save import SaveExperiment
    with open('experiment.json', mode='r') as f:
        experiment = json.loads(f.read())
    dt = DatasetFactory(name=experiment['dataset'], concat=True)

    ldr = LoaderState(
        id_exp='cifar10_resnet_k10',
        epochs=experiment['epochs'],
        dataset=dt,
        valid_exp=experiment['exp'],
        url=experiment['dir'],
    )
    state = ldr.state
    state.get_state_validation('kfold', k=10).status = True
    state.get_state_validation('kfold', k=10).weights[0] = 'teste'
    print(state.get_state_validation('kfold', k=10).weights)
    v = state.get_validation('kfold')
    print(v)
    SaveExperiment(root_dir=experiment['dir'] + 'states/').save_state(state.to_dict())

    
    # print(ldr.state.valid_exp['kfold'][0].to_dict())

    
    
    
    