import numpy as np
from neural_pipeline import MetricsProcessor, MetricsGroup, AbstractMetric
from torch import Tensor
from sklearn.metrics import roc_auc_score

__all__ = ['ROCAUCMetric', 'ClassificationMetricsProcessor']


class ROCAUCMetric(AbstractMetric):
    def __init__(self, thresold: float, name: str = 'ROC_AUC'):
        super().__init__(name)
        self._thresh = thresold
        self._targets, self._preds = [], []
        self._pred_preprocess = lambda x: x.data.cpu().numpy()
        self._target_preprocess = lambda x: x.data.cpu().numpy()

    def set_pred_preproc(self, preproc: callable) -> 'ROCAUCMetric':
        self._pred_preprocess = preproc
        return self

    def set_target_preproc(self, preproc: callable) -> 'ROCAUCMetric':
        self._target_preprocess = preproc
        return self

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

        pred[pred < self._thresh] = 0
        self._preds.extend(list(pred))
        self._targets.extend(list(tar))
        return 0

    def get_values(self):
        res = roc_auc_score(np.squeeze(self._targets), np.squeeze(self._preds))
        self._targets, self._preds = [], []
        return res

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
            np.ndarray of shape [B, 1]
        """
        val_internal = val.data.cpu().numpy()
        idx = np.argmax(val_internal, axis=1)
        max_vals = val_internal[np.arange(len(val_internal)), idx]
        return np.where(idx > 0, max_vals, 1 - max_vals)

    @staticmethod
    def multiclass_target_preproc(val: Tensor):
        """
        Multiclass target preprocess method.

        Args:
              val (Tensor): values to target as Tensor of size [B, 1]

        Returns:
            np.ndarray of shape [B, 1]
        """
        val_internal = val.data.cpu().numpy()
        return (val_internal > 0).astype(np.int)


class ClassificationMetricsProcessor(MetricsProcessor):
    def __init__(self, name: str, thresholds: [float]):
        super().__init__()

        self._auc_metrics = []
        auc_group = MetricsGroup('ROC_AUC')
        for thresh in thresholds:
            self._auc_metrics.append(auc_group.add(ROCAUCMetric(thresh, '{}_{}'.format(name, thresh))))

        self.add_metrics_group(auc_group)

    def set_pred_preproc(self, preproc: callable) -> 'ClassificationMetricsProcessor':
        for m in self._auc_metrics:
            m.set_pred_preproc(preproc)
        return self

    def set_target_preproc(self, preproc: callable) -> 'ClassificationMetricsProcessor':
        for m in self._auc_metrics:
            m.set_target_preproc(preproc)
        return self
