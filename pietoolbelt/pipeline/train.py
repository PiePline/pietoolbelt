from typing import List

from piepline.train import Trainer

from pietoolbelt.pipeline.abstract_step import AbstractStep, AbstractStepResult


class Train(AbstractStep):
    def __init__(self, trainer: Trainer, output_res: AbstractStepResult, input_res: List[AbstractStepResult] = None):
        super().__init__(output_res=output_res, input_results=input_res)
        self._trainer = trainer

    def run(self):
        self._trainer.train()
