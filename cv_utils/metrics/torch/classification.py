import numpy as np
from neural_pipeline import MetricsProcessor, MetricsGroup, AbstractMetric
from torch import Tensor
from sklearn.metrics import roc_auc_score


class AUCMetric(AbstractMetric):
    def __init__(self, name: str, thresold: float):
        super().__init__(name)
        self._thresh = thresold
        self._targets, self._preds = [], []

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        out = output.data.cpu().numpy()[:, 1]
        tar = target.data.cpu().numpy()

        out[out < self._thresh] = 0
        self._preds.extend(list(out))
        self._targets.extend(list(np.argmax(tar, axis=1)))
        return 0

    def get_values(self):
        try:
            res = roc_auc_score(self._targets, self._preds)
        except:
            res = 0

        self._targets, self._preds = [], []
        return res


class ClassificationMetricsProcessor(MetricsProcessor):
    def __init__(self, name: str, thresholds: [float]):
        super().__init__()

        auc_group = MetricsGroup('AUC')
        for thresh in thresholds:
            auc_group.add(AUCMetric('{}_{}'.format(name, thresh), thresh))

        self.add_metrics_group(auc_group)
