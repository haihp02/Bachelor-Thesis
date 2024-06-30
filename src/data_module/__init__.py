class DataModule(object):
    """
    Base class 
    """

    def __init__(self):
        pass

    def get_dataloader(self, mode):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(mode='train')
    
    def val_dataloader(self):
        return self.get_dataloader(mode='val')
    
    def test_dataloader(self):
        return self.get_dataloader(mode='test')
    
    def get_tensorloader(self):
        pass