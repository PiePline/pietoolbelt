import unittest
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
import numpy as np

from pietoolbelt.metrics.torch.classification import ROCAUCMetric

__all__ = ['PyTorchTest']


class PyTorchTest(unittest.TestCase):
    def test_standard_case(self):
        for _ in range(10):
            out = torch.rand(10, 1)
            target = torch.ones(10, 1)
            target[0][0] = 0
            thresh = 0.5

            out_for_res = np.where(out.numpy() < thresh, 0, 1)
            true_res = roc_auc_score(target.numpy().astype(np.int), out_for_res)

            metric = ROCAUCMetric(thresh)
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
        for _ in range(10):
            pred = torch.rand(10, 4)
            target = torch.from_numpy((np.random.rand(10, 1) * 4).astype(np.int))
            thresh = 0.7

            metric = ROCAUCMetric(thresh)\
                .set_pred_preproc(ROCAUCMetric.multiclass_pred_preproc)\
                .set_target_preproc(ROCAUCMetric.multiclass_target_preproc)
            metric.calc(pred, target)
            res = metric.get_values()

            if target.max() < 1 or target.min() > 0:
                self.assertIs(res, np.nan)
            else:
                out_for_res = ROCAUCMetric.multiclass_pred_preproc(pred)
                out_for_res = np.where(out_for_res < thresh, 0, 1)
                true_res = roc_auc_score(ROCAUCMetric.multiclass_target_preproc(target), out_for_res)

                self.assertTrue(np.allclose(res, true_res), msg="res: {}, true_res: {}".format(res, true_res))
