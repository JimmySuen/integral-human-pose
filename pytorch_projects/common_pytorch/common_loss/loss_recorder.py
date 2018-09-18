import torch
class LossRecorder(object):
    def __init__(self):
        self.sum = 0
        self.count = 0.0000001

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def get_avg(self):
        mean_val = self.sum / self.count
        if isinstance(mean_val, torch.Tensor):
            return mean_val.item()
        else:
            return mean_val

    def get_name_value(self):
        return [("Loss", self.sum / self.count)]

    def reset(self):
        self.sum = 0
        self.count = 0.0000001