import unittest
import unittest.mock

import numpy as np

import hpolib.benchmarks.ml.autosklearn_benchmark


class TestAutoSklearnBenchmark(unittest.TestCase):

    @unittest.mock.patch.multiple(hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark, __abstractmethods__=set())
    @unittest.mock.patch('hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark._check_dependencies')
    @unittest.mock.patch('hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark._get_data_manager')
    @unittest.mock.patch('hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark._setup_backend')
    @unittest.mock.patch('hpolib.abstract_benchmark.AbstractBenchmark.__init__')
    def test_init(self, super_init_mock, setup_backend_mock,
                  get_data_manager_mock, check_dependencies_mock):
        fixture = 'sentinel'
        get_data_manager_mock.return_value = fixture
        auto = hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark(1)
        self.assertEqual(auto.data_manager, fixture)
        self.assertEqual(super_init_mock.call_count, 1)
        self.assertEqual(setup_backend_mock.call_count, 1)
        self.assertEqual(get_data_manager_mock.call_count, 1)
        self.assertEqual(check_dependencies_mock.call_count, 1)

class TestIntegration(unittest.TestCase):

    def test_multiclass_on_iris(self):
        auto = hpolib.benchmarks.ml.autosklearn_benchmark.MulticlassClassificationBenchmark(289)
        all_rvals = []

        for i in range(10):
            print(i)
            train_rval, test_rval = auto.test(1, fold=i)
            for r in train_rval:
                print(r)
                all_rvals.append(r['function_value'])
            for r in test_rval:
                all_rvals.append(r['function_value'])

        self.assertLess(np.mean(all_rvals), 1.0)
        self.assertGreater(np.mean(all_rvals), 0.0)
        self.assertGreaterEqual(np.max(all_rvals), 0.0)
        self.assertLessEqual(np.max(all_rvals), 2.0)
        self.assertEqual(len(all_rvals), 20)
