import logging
import tempfile
import time
import autosklearn
import autosklearn.evaluation
import autosklearn.data.xy_data_manager
import autosklearn.util.backend
import autosklearn.constants
import autosklearn.util.pipeline
import numpy as np
import openml.tasks
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.dependencies import verify_packages

__version__ = 0.1


class AutoSklearnBenchmark(AbstractBenchmark):
    """Base class for auto-sklearn benchmarks.

    auto-sklearn benchmarks implement Section 6 of the paper 'Efficient and
    Robust Automated Machine Learning' by Feurer et al., published in
    Proceedings of NIPS 2015.
    """

    def __init__(self, task_id):
        super().__init__()
        self._check_dependencies()
        self.data_manager = self._get_data_manager(task_id)
        self._setup_evaluators(self.data_manager)

    def _setup_evaluators(self, data_manager):
        tmp_folder = tempfile.mkdtemp()
        output_folder = tempfile.mkdtemp()
        self.backend = autosklearn.util.backend.create(
            temporary_directory=tmp_folder,
            output_directory=output_folder,
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True)
        self.backend.save_datamanager(data_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.evaluator = autosklearn.evaluation.ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            resampling_strategy='partial-cv',
            folds=10,
            logger=self.logger)
        self.test_evaluator = autosklearn.evaluation.ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            resampling_strategy='test',
            logger=self.logger)

    @staticmethod
    def _get_data_manager(task_id):

        task = openml.tasks.get_task(task_id)

        try:
            task.get_train_test_split_indices(fold=1, repeat=0)
            raise_exception = True
        except:
            raise_exception = False
        if raise_exception:
            raise ValueError('Task %d has more than one fold. This benchmark '
                             'can only work with a single fold.' % task_id)
        try:
            task.get_train_test_split_indices(fold=0, repeat=1)
            raise_exception = True
        except:
            raise_exception = False
        if raise_exception:
            raise ValueError('Task %d has more than one repeat. This benchmark '
                             'can only work with a single repeat.' % task_id)

        train_indices, test_indices = task.get_train_test_split_indices()

        X, y = task.get_X_and_y()
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        num_classes = len(np.unique(y_train))
        if num_classes == 2:
            task_type = autosklearn.constants.BINARY_CLASSIFICATION
        elif num_classes > 2:
            task_type = autosklearn.constants.MULTICLASS_CLASSIFICATION
        else:
            raise ValueError('This benchmark needs at least two classes.')

        dataset = task.get_dataset()
        _, _, categorical_indicator = dataset.get_data(
            target=task.target_name,
            return_categorical_indicator=True)
        variable_types = ['categorical' if ci else 'numerical'
                          for ci in categorical_indicator]

        # TODO in the future, the XYDataManager should have this as it's own
        # attribute
        data_manager = autosklearn.data.xy_data_manager.XYDataManager(
            X_train, y_train, task_type, autosklearn.constants.BAC_METRIC,
            variable_types, dataset.name, False)
        data_manager._data['X_test'] = X_test
        data_manager._data['y_test'] = y_test

        return data_manager

    def _check_dependencies(self):
        dependencies = ['numpy>=1.9.0',
                        'scipy>=0.14.1',
                        'scikit-learn==0.17.1',
                        'xgboost==0.4a30',
                        'pynisher==0.4.2',
                        'auto-sklearn==0.1.1']
        dependencies = '\n'.join(dependencies)
        verify_packages(dependencies)

    @staticmethod
    def get_meta_information():
        return {'bibtex': """@incollection{NIPS2015_5872,
title = {Efficient and Robust Automated Machine Learning},
author = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina and Springenberg, Jost and Blum, Manuel and Hutter, Frank},
booktitle = {Advances in Neural Information Processing Systems 28},
editor = {C. Cortes and N. D. Lawrence and D. D. Lee and M. Sugiyama and R. Garnett},
pages = {2962--2970},
year = {2015},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf}
}"""}

    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration, **kwargs):

        start_time = time.time()

        fold = kwargs['fold']
        cutoff = kwargs.get('cutoff', 1800)
        memory_limit = kwargs.get('memory_limit', 3072)

        status, cost, runtime, additional_run_info = self.evaluator.run(
            config=configuration, cutoff=cutoff, memory_limit=memory_limit,
            instance=fold)

        end_time = time.time()

        return {'function_value': cost, 'cost': end_time - start_time}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, configuration, **kwargs):
        start_time = time.time()

        cutoff = kwargs.get('cutoff', 3600)
        memory_limit = kwargs.get('memory_limit', 6144)

        status, cost, runtime, additional_run_info = self.test_evaluator.run(
            config=configuration, cutoff=cutoff, memory_limit=memory_limit)

        end_time = time.time()

        return {'function_value': cost, 'cost': end_time - start_time}


class MulticlassClassificationBenchmark(AutoSklearnBenchmark):
    @staticmethod
    def get_configuration_space():
        task = autosklearn.constants.MULTICLASS_CLASSIFICATION
        metric = autosklearn.constants.BAC_METRIC
        cs = autosklearn.util.pipeline.get_configuration_space(
            info={'task': task, 'metric': metric, 'is_sparse': 0})
        return cs


class AutoSklearnBenchmarkAdultBAC(MulticlassClassificationBenchmark):
    def __init__(self):
        super().__init__(2117)


if __name__ == '__main__':
    benchmark = AutoSklearnBenchmarkAdultBAC()
    benchmark.test(5)
