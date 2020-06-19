from tensorflow.keras.optimizers import SGD, Adam

class Optimizers:
    def __init__(self, opt, params):
        self.opt = None
        if opt == 'sgd':
            if not params:
                self.opt = SGD()
            else:
                self.opt = SGD(learning_rate=params['learning_rate'],
                        momentum=params['momentum'],
                        decay=params['decay'])
        elif opt == 'adam':
            self.opt = Adam()

    def optimizer(self):
        return self.opt