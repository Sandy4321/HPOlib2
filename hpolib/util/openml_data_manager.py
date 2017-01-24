import logging
import os

import numpy as np
import openml
from sklearn.model_selection import train_test_split

import hpolib
from hpolib.util.data_manager import HoldoutDataManager, \
    CrossvalidationDataManager


def _load_data(task_id):
    task = openml.tasks.get_task(task_id)
    try:
        task.get_train_test_split_indices(fold=1, repeat=0)
        raise_exception = True
    except:
        raise_exception = False

    if raise_exception:
        raise ValueError('Task %d has more than one fold. This benchmark '
                         + 'can only work with a single fold.' % task_id)

    try:
        task.get_train_test_split_indices(fold=0, repeat=1)
        raise_exception = True
    except:
        raise_exception = False

    if raise_exception:
        raise ValueError('Task %d has more than one repeat. This benchmark '
                         + 'can only work with a single repeat.' % task_id)

    train_indices, test_indices = task.get_train_test_split_indices()

    X, y = task.get_X_and_y()

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


class OpenMLHoldoutDataManager(HoldoutDataManager):
    def __init__(self, openml_task_id, rng=None):

        super().__init__()

        self.save_to = os.path.join(hpolib._config.data_dir, "OpenML")
        self.task_id = openml_task_id

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory %s", self.save_to)
            os.makedirs(self.save_to)

        openml.config.apikey = '953f6621518c13791dbbfc6d3698f5ad'
        openml.config.cachedir = self.save_to

        super().__init__()

    def load(self):
        """
        Loads dataset from OpenML in _config.data_directory.
        Downloads data if necessary.
        """

        X_train, y_train, X_test, y_test = _load_data(self.task_id)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                              test_size=0.33,
                                                              random_state=self.rng)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_valid
        self.y_val = y_valid
        self.X_test = X_test
        self.y_test = y_test

        return self.X_train, self.y_train, self.X_val, self.y_val, \
               self.X_test, self.y_test


class OpenMLCrossvalidationDataManager(CrossvalidationDataManager):
    def __init__(self, openml_task_id, rng=None):

        super().__init__()

        self.save_to = os.path.join(hpolib._config.data_dir, "OpenML")
        self.task_id = openml_task_id

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory %s", self.save_to)
            os.makedirs(self.save_to)

        openml.config.apikey = '953f6621518c13791dbbfc6d3698f5ad'
        openml.config.cachedir = self.save_to

        super().__init__()

    def load(self):
        """
        Loads dataset from OpenML in _config.data_directory.
        Downloads data if necessary.
        """

        X_train, y_train, X_test, y_test = _load_data(self.task_id)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        return self.X_train, self.y_train, self.X_test, self.y_test