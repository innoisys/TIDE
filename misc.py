import sys
import pickle as pckl
import os
import tensorflow as tf
import pathlib

def get_arg(name='', default=None):
    name = "--%s=" % name
    for item in sys.argv:
        if item.startswith(name):
            return item.replace(name, '')
    return default


def cache(cache_file, default=None):
    data = default
    if not os.path.exists(cache_file):
        if callable(default):
            data = default()
        if not os.path.exists(os.path.dirname(cache_file)):
            os.mkdir(os.path.dirname(cache_file))
        with open(cache_file, 'wb') as file:
            pckl.dump(data, file)
    else:
        with open(cache_file, 'rb') as file:
            data = pckl.load(file)
    return data


def ensure_path(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_interval=1, path='{epoch}-model.hdf5'):
        super(CheckpointCallback, self).__init__()
        self.epoch_interval = epoch_interval
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_interval == 0 and epoch > 0:
            save_path = self.path
            if 'epoch' in save_path:
                save_path = save_path.format(epoch=epoch)
            self.model.save_weights(save_path)

