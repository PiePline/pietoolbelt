import torch
from piepline.train import Trainer
from piepline.train_config.train_config import BaseTrainConfig

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult, AbstractStep, DatasetInPipeline


class TrainResult(AbstractStepDirResult):
    def __init__(self, path: str):
        super().__init__(path=path)


class PipelineTrainer(Trainer, AbstractStep):
    def __init__(self, train_config: BaseTrainConfig, output_res: TrainResult, dataset: DatasetInPipeline,
                 device: str = None):
        Trainer.__init__(self, train_config, device=torch.device(device) if device is not None else None)
        AbstractStep.__init__(self, output_res=output_res, input_results=[dataset])

    def run(self):
        self.train()
