import time
import logging
import numpy as np

import re
import matplotlib.pyplot as plt


class TensorboardCallback(object):
    """Log metrics periodically in TensorBoard.
    This callback works almost same as `callback.Speedometer`, but write TensorBoard event file
    for visualization. For more usage, please refer https://github.com/dmlc/tensorboard
    Parameters
    ----------
    logging_dir : str
        TensorBoard event file directory.
        After that, use `tensorboard --logdir=path/to/logs` to launch TensorBoard visualization.
    prefix : str
        Prefix for a metric name of `scalar` value.
        You might want to use this param to leverage TensorBoard plot feature,
        where TensorBoard plots different curves in one graph when they have same `name`.
        The follow example shows the usage(how to compare a train and eval metric in a same graph).
    Examples
    --------
    >>> # log train and eval metrics under different directories.
    >>> training_log = 'logs/train'
    >>> evaluation_log = 'logs/eval'
    >>> # in this case, each training and evaluation metric pairs has same name,
    >>> # you can add a prefix to make it separate.
    >>> batch_end_callbacks = [mx.contrib.tensorboard.LogMetricsCallback(training_log)]
    >>> eval_end_callbacks = [mx.contrib.tensorboard.LogMetricsCallback(evaluation_log)]
    >>> # run
    >>> model.fit(train,
    >>>     ...
    >>>     batch_end_callback = batch_end_callbacks,
    >>>     eval_end_callback  = eval_end_callbacks)
    >>> # Then use `tensorboard --logdir=logs/` to launch TensorBoard visualization.
    """
    def __init__(self, logging_dir, total_step=0, prefix=None):
        self.prefix = prefix
        self.step = total_step
        try:
            from tensorboardX import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param, name_value=None):
        """Callback to log training speed and metrics in TensorBoard."""
        if name_value:
            self._add_scalar(name_value)

        if param.loss_metric is None:
            return

        if param.add_step:
            self.step += 1

        name_value = param.loss_metric.get_name_value()
        if name_value:
            self._add_scalar(name_value)

    def _add_scalar(self, name_value):
        for name, value in name_value:
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_scalar(name, value, self.step)