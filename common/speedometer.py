import time
import logging
from collections import namedtuple


BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'total_batch',
                            'add_step',
                            'eval_metric',
                            'loss_metric',
                            'locals'])
# speed, eval and loss metric
class Speedometer(object):
    def __init__(self, batch_size, frequent=50, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        total = param.total_batch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                msg = 'Epoch[%d] Batch [%d]in[%d]\tSpeed: %.2f samples/sec'%(param.epoch, count, total, speed)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                    for (name, value) in name_value:
                        msg += '\t%s=%f'%(name, value)
                if param.loss_metric is not None:
                    name_value = param.loss_metric.get_name_value()
                    if self.auto_reset:
                        param.loss_metric.reset()
                    for (name, value) in name_value:
                        msg += '\t%s=%f'%(name, value)
                logging.info(msg)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()