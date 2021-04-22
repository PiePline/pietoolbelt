from typing import List

import numpy as np
from piepline.train_config.metrics import MetricsGroup, AbstractMetric

import mlflow

from piepline.monitoring.monitors import AbstractMetricsMonitor, AbstractLossMonitor


class MLFlowMonitor(AbstractMetricsMonitor, AbstractLossMonitor):
    def __init__(self, server_url: str, project_name: str):
        AbstractMetricsMonitor.__init__(self)
        AbstractLossMonitor.__init__(self)

        mlflow.set_tracking_uri(server_url)
        mlflow.set_experiment(project_name)

    def update_losses(self, losses: {}) -> None:
        """
        Update monitor

        :param losses: losses values with keys 'train' and 'validation'
        """

        def on_loss(name: str, values: np.ndarray or dict) -> None:
            if isinstance(values, dict):
                mlflow.log_metrics({'loss_{}/k'.format(name, k): np.mean(v) for k, v in values.items()}, step=self._epoch_num)
            else:
                mlflow.log_metric('loss/{}'.format(name), np.mean(values), step=self._epoch_num)

        self._iterate_by_losses(losses, on_loss)

    def _process_metric(self, path: List[MetricsGroup], metric: AbstractMetric):
        """
        Update console

        :param metrics: metrics
        """

        tag = '/'.join([p.name() for p in path] + [metric.name()])

        values = metric.get_values().astype(np.float32)
        if values.size > 0:
            mlflow.log_metric(tag, float(np.mean(values)), step=self._epoch_num)

    def update_scalar(self, name: str, value: float, epoch_idx: int = None) -> None:
        """
        Update scalar on tensorboard

        :param name: the classic tag for TensorboardX
        :param value: scalar value
        :param epoch_idx: epoch idx. If doesn't set - use last epoch idx stored in this class
        """
        mlflow.log_metric(name, value, step=epoch_idx)

    def close(self):
        mlflow.end_run()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
