class Trainer():
    """The base class for training models with data"""

    def __init__(self, args):
        self.args = args

    def prepare(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError
    
    