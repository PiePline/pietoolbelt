import json
import os
import unittest

import numpy as np
import shutil

__all__ = ['StratificationTest']

from pietoolbelt.datasets.common import BasicDataset
from pietoolbelt.pipeline.stratification import DatasetStratification, StratificationResult


class _DatasetMock(BasicDataset):
    def __init__(self):
        items = list(range(100))
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        return {'data': item}


class StratificationResultTest(unittest.TestCase):
    RESULT_DIR = 'tmp_result_dir'

    def tearDown(self):
        if os.path.exists(StratificationResultTest.RESULT_DIR):
            shutil.rmtree(StratificationResultTest.RESULT_DIR, ignore_errors=True)

    def test_init(self):
        try:
            StratificationResult(path=StratificationResultTest.RESULT_DIR)
        except TypeError as err:
            self.fail("Can't instantiate StratificationResult class. Bad arguments")
        except ...:
            self.fail("Can't instantiate StratificationResult class")

    def test_dir_creation(self):
        StratificationResult(path=StratificationResultTest.RESULT_DIR)
        self.assertTrue(os.path.exists(StratificationResultTest.RESULT_DIR))

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


class StratificationTest(unittest.TestCase):
    def test_init(self):
        try:
            DatasetStratification(dataset=_DatasetMock(), calc_target_label=lambda x: x['data'] // 10,
                                  result=StratificationResult('tmp'), workers_num=2)
        except TypeError as err:
            self.fail("Can't instantiate DatasetStratification class. Bad arguments")
        except ...:
            self.fail("Can't instantiate DatasetStratification class")
