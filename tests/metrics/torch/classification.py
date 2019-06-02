import unittest
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
import numpy as np

from cv_utils.metrics.torch.classification import ROCAUCMetric

__all__ = ['PyTorchTest']


class PyTorchTest(unittest.TestCase):
    def test_standard_case(self):
        out = torch.rand(10, 1)
        target = torch.ones(10, 1)
        target[0][0] = 0

        true_res = roc_auc_score(target.numpy().astype(np.int), out.numpy())

        metric = ROCAUCMetric(0.5)
        metric.calc(out, target)
        res = metric.get_values()

        self.assertTrue(np.allclose(res, true_res), msg="res: {}, true_res: {}".format(res, true_res))

    def test_builtin_preproc_methods(self):
        out = torch.Tensor([[0.1, 0.5, 0.3],
                            [0.9, 0.89, 0.1],
                            [0.9, 0.3, 0.99],
                            [0.1, 0.4, 0.3]])

        res = ROCAUCMetric.multiclass_pred_preproc(out)
        self.assertTrue(np.allclose(res, [0.5, 0.1, 0.99, 0.4]))

        target = torch.Tensor([[0], [1], [3], [4], [0]])
        res = ROCAUCMetric.multiclass_target_preproc(target)
        self.assertTrue(np.allclose(res, [[0], [1], [1], [1], [0]]))

    def test_multiclass_case(self):
        pred = torch.rand(10, 4)
        target = torch.from_numpy((np.random.rand(10, 1) * 4).astype(np.int))

        true_res = roc_auc_score(ROCAUCMetric.multiclass_target_preproc(target), ROCAUCMetric.multiclass_pred_preproc(pred))

        metric = ROCAUCMetric(0.5)\
            .set_pred_preproc(ROCAUCMetric.multiclass_pred_preproc)\
            .set_target_preproc(ROCAUCMetric.multiclass_target_preproc)
        metric.calc(pred, target)
        res = metric.get_values()

        self.assertTrue(np.allclose(res, true_res), msg="res: {}, true_res: {}".format(res, true_res))
