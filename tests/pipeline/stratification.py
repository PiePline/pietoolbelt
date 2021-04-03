import json
import os
import unittest
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
import shutil

__all__ = ['StratificationResultTest', 'DatasetStratificationTest', 'TestDatasetStratificationInPipeline']

from pietoolbelt.datasets.common import BasicDataset
from pietoolbelt.pipeline.abstract_step import DatasetInPipeline
from pietoolbelt.pipeline.stratification import DatasetStratification, StratificationResult, PipelineDatasetStratification


class _DatasetMock(BasicDataset):
    def __init__(self):
        items = list(range(100))
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        return {'data': item}


class _BaseTest(unittest.TestCase):
    RESULT_DIR = 'tmp_result_dir'

    def tearDown(self):
        if os.path.exists(_BaseTest.RESULT_DIR):
            shutil.rmtree(_BaseTest.RESULT_DIR, ignore_errors=True)


def _calc_label(x):
    return x // 10


class StratificationResultTest(_BaseTest):
    def test_init(self):
        try:
            StratificationResult(path=StratificationResultTest.RESULT_DIR)
        except TypeError as err:
            self.fail("Can't instantiate StratificationResult class. Bad arguments")
        except ...:
            self.fail("Can't instantiate StratificationResult class")

    def test_dir_creation(self):
        result = StratificationResult(path=StratificationResultTest.RESULT_DIR)
        self.assertTrue(os.path.exists(StratificationResultTest.RESULT_DIR))
        self.assertEqual(result.get_output_paths(), [StratificationResultTest.RESULT_DIR])

        with self.assertRaises(Exception):
            StratificationResult(path=StratificationResultTest.RESULT_DIR)

        with self.assertRaises(Exception):
            StratificationResult(path=StratificationResultTest.RESULT_DIR, allow_exist=False)

        try:
            StratificationResult(path=StratificationResultTest.RESULT_DIR, allow_exist=True)
        except ...:
            self.fail("Cant instantiate StratificationResult when dir exists and existing are allowed")

    def test_indices_flushing_and_loading(self):
        result = StratificationResult(path=StratificationResultTest.RESULT_DIR)

        indices_dict = {'a': (np.random.randn(10) * 100).astype(np.uint8),
                        'bbbb': (np.random.randn(20) * 100).astype(np.uint8),
                        'ccccc': (np.random.randn(30) * 100).astype(np.uint8)}
        for name, indices in indices_dict.items():
            result.add_indices(indices=indices, name=name, dataset=_DatasetMock())
            self.assertTrue(os.path.exists(os.path.join(StratificationResultTest.RESULT_DIR, name + '.npy')))

        meta_file_path = os.path.join(StratificationResultTest.RESULT_DIR, 'meta.json')
        self.assertTrue(os.path.exists(meta_file_path))

        with open(os.path.join(StratificationResultTest.RESULT_DIR, 'meta.json'), 'r') as meta_file:
            meta = json.load(meta_file)
        self.assertEqual(meta, {"a": {"indices_num": 10}, "bbbb": {"indices_num": 20}, "ccccc": {"indices_num": 30}})

        new_result = StratificationResult(path=StratificationResultTest.RESULT_DIR, allow_exist=True)

        for name, indices in indices_dict.items():
            loaded_indices = new_result.get_indices(name)
            self.assertEqual(indices.tolist(), loaded_indices.tolist())


class _BaseStratificationTest(_BaseTest, metaclass=ABCMeta):
    @abstractmethod
    def init_stratificator(self):
        return None

    def test_init(self):
        try:
            self.init_stratificator()
        except TypeError as err:
            self.fail("Can't instantiate DatasetStratification class. Bad arguments")
        except Exception as err:
            self.fail("Can't instantiate DatasetStratification class")

    def _test_stratification(self, stratification):
        def test_indices(path: str, target_num: int):
            self.assertTrue(os.path.exists(path))
            self.assertEqual(target_num, len(np.load(path)))

        stratification.run(parts={'a': 0.3, 'b': 0.7})
        test_indices(os.path.join(_BaseStratificationTest.RESULT_DIR, 'a.npy'), target_num=30)
        test_indices(os.path.join(_BaseStratificationTest.RESULT_DIR, 'b.npy'), target_num=70)

        indices_1 = np.load(os.path.join(_BaseStratificationTest.RESULT_DIR, 'a.npy'))
        indices_2 = np.load(os.path.join(_BaseStratificationTest.RESULT_DIR, 'b.npy'))

        self.assertTrue(np.isin(indices_1, indices_2).max() == 0)

        stratification.run(parts={'a': 0.7, 'b': 0.3})
        test_indices(os.path.join(_BaseStratificationTest.RESULT_DIR, 'a.npy'), target_num=70)
        test_indices(os.path.join(_BaseStratificationTest.RESULT_DIR, 'b.npy'), target_num=30)

        with self.assertRaises(RuntimeError):
            stratification.run(parts={'a': 0.6, 'b': 0.7})

    def test_stratification(self):
        stratification = DatasetStratification(dataset=_DatasetMock(), calc_target_label=lambda x: x // 10,
                                               result=StratificationResult(_BaseStratificationTest.RESULT_DIR), workers_num=0)
        self._test_stratification(stratification)

    def test_multiprocess_stratification(self):
        multiprocess_stratification = DatasetStratification(dataset=_DatasetMock(), calc_target_label=_calc_label,
                                                            result=StratificationResult(_BaseStratificationTest.RESULT_DIR),
                                                            workers_num=2)
        self._test_stratification(multiprocess_stratification)


class DatasetStratificationTest(_BaseStratificationTest):
    def init_stratificator(self):
        DatasetStratification(dataset=_DatasetMock(), calc_target_label=lambda x: x // 10,
                              result=StratificationResult(_BaseStratificationTest.RESULT_DIR), workers_num=2)


class _DatasetInPipelineMock(DatasetInPipeline):
    def __init__(self):
        items = list(range(100))
        super().__init__(items)

    def get_output_paths(self) -> List[str]:
        return ['fake_dir']

    def _interpret_item(self, item) -> any:
        return {'data': item}


class TestDatasetStratificationInPipeline(_BaseStratificationTest):
    def init_stratificator(self):
        dataset = _DatasetInPipelineMock()
        result = StratificationResult(TestDatasetStratificationInPipeline.RESULT_DIR)
        stratification = PipelineDatasetStratification(dataset=dataset, calc_target_label=lambda x: x // 10, result=result,
                                                       workers_num=0)
        return stratification

    def test_interface(self):
        dataset = _DatasetInPipelineMock()
        result = StratificationResult(TestDatasetStratificationInPipeline.RESULT_DIR)
        stratification = PipelineDatasetStratification(dataset=dataset, calc_target_label=lambda x: x // 10, result=result,
                                                       workers_num=0)

        self.assertEqual(stratification.get_input_results(), [dataset])
        self.assertEqual(stratification.get_output_res(), result)
