from abc import abstractmethod, ABCMeta

import numpy as np
from piepline import MetricsProcessor, MetricsGroup, AbstractMetric
from torch import Tensor
from sklearn.metrics import roc_auc_score, confusion_matrix

__all__ = ['ROCAUCMetric', 'ClassificationMetricsProcessor']


class _ClassificationMetric(AbstractMetric):
    def __init__(self, name: str):
        super().__init__(name)

        self._targets, self._preds = [], []
        self._pred_preprocess = lambda x: x.data.cpu().numpy()
        self._target_preprocess = lambda x: x.data.cpu().numpy()

    def set_pred_preproc(self, preproc: callable) -> '_ClassificationMetric':
        self._pred_preprocess = preproc
        return self

    def set_target_preproc(self, preproc: callable) -> '_ClassificationMetric':
        self._target_preprocess = preproc
        return self

    @staticmethod
    def multiclass_pred_preproc(val: Tensor):
        """
        Multiclass predict preprocess method.

        Method choose index of max element. If index == 0, than return 1-val[index], otherwise val[index].

        For example:
        ```
            pred = torch.Tensor([[0.1, 0.5, 0.3],
                                 [0.9, 0.89, 0.1],
                                 [0.9, 0.3, 0.99],
                                 [0.1, 0.4, 0.3]])

            res = ROCAUCMetric.multiclass_pred_preproc(pred)
            res: [0.5, 0.1, 0.99, 0.4]
        ```

        Args:
              val (Tensor): values to preprocess as Tensor of size [B, C]

        Returns:
            np.ndarray of shape [B]
        """
        val_internal = val.data.cpu().numpy()
        idx = np.argmax(val_internal, axis=1)
        max_vals = val_internal[np.arange(len(val_internal)), idx]
        return np.squeeze(np.where(idx > 0, max_vals, 1 - max_vals))

    @staticmethod
    def multiclass_target_preproc(val: Tensor):
        """
        Multiclass target preprocess method.

        Args:
              val (Tensor): values to target as Tensor of size [B, 1]

        Returns:
            np.ndarray of shape [B]
        """
        val_internal = val.data.cpu().numpy()
        return np.squeeze(np.clip(val_internal, 0, 1).astype(np.int))

    def calc(self, predict: Tensor, target: Tensor) -> np.ndarray or float:
        """
        Calc metric

        Args:
            predict (Tensor): predict classes as Tensor of size [B, C]
            target (Tensor): ground truth classes as Tensor of size [B, C]
        Returns:
             return zero cause metric accumulate all values and calc when :meth:`get_values`
        """
        pred = self._pred_preprocess(predict)
        tar = self._target_preprocess(target)

        self._preds.extend(pred)
        self._targets.extend(tar)
        return 0

    def _calc(self, output: Tensor, target: Tensor):
        self.calc(output, target)

    @abstractmethod
    def _get_values(self):
        pass

    def get_values(self):
        """
        Get values of metric
        :return:
        """
        res = self._get_values()
        self._targets, self._preds = [], []
        return np.array([res]) if type(res) is float else res


class ROCAUCMetric(_ClassificationMetric):
    def __init__(self, thresold: float, name: str = 'ROC_AUC'):
        super().__init__(name)
        self._thresh = thresold

    def _get_values(self):
        preds = np.where(np.squeeze(self._preds) < self._thresh, 0, 1)

        try:
            res = roc_auc_score(np.squeeze(self._targets), preds)
        except ValueError:
            return np.nan

        return res


class RecallMetric(_ClassificationMetric):
    def __init__(self, threshold: float, name: str = "Recall"):
        super().__init__(name)
        self._thresh = threshold

    def _get_values(self):
        preds = np.where(np.squeeze(self._preds) < self._thresh, 0, 1)
        tn, fp, fn, tp = confusion_matrix(np.squeeze(self._targets), preds).ravel()
        return tp / (tp + fn)


class ActCMetric(_ClassificationMetric):
    def __init__(self, threshold: float, name: str = "ActC"):
        super().__init__(name)
        self._thresh = threshold

    def _get_values(self):
        preds = np.where(np.squeeze(self._preds) < self._thresh, 0, 1)
        tn, fp, fn, tp = confusion_matrix(np.squeeze(self._targets), preds).ravel()
        return fp / (fp + tn) + 19 * fn / (fn + tp)


class ClassificationMetricsProcessor(MetricsProcessor):
    def __init__(self, name: str, thresholds: [float]):
        super().__init__()

        self._auc_metrics = []
        auc_group = MetricsGroup('ROC_AUC')
        if thresholds is None:
            self._auc_metrics.append(ROCAUCMetric(0.5, name))
            auc_group.add(self._auc_metrics[-1])
        else:
            for thresh in thresholds:
                self._auc_metrics.append(ROCAUCMetric(thresh, '{}_{}'.format(name, thresh)))
                auc_group.add(self._auc_metrics[-1])

        self.add_metrics_group(auc_group)

    def set_pred_preproc(self, preproc: callable) -> 'ClassificationMetricsProcessor':
        for m in self._auc_metrics:
            m.set_pred_preproc(preproc)
        return self

    def set_target_preproc(self, preproc: callable) -> 'ClassificationMetricsProcessor':
        for m in self._auc_metrics:
            m.set_target_preproc(preproc)
        return self
