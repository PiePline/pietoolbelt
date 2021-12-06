import json
import os
from typing import List, Callable, Dict

from piepline.train import Trainer
from piepline.utils.fsm import FileStructManager

from pietoolbelt.pipeline.abstract_step import AbstractStep, AbstractStepDirResult
from pietoolbelt.pipeline.stratification import StratificationResult

__all__ = ['FoldedTrainResult', 'FoldedTrainer', 'PipelineFoldedTrainer']


class FoldedTrainResult(AbstractStepDirResult):
    def __init__(self, path: str):
        super().__init__(path)
        self._folds = dict()
        self._meta_file_path = os.path.join(self._path, 'meta.json')

        if os.path.exists(self._meta_file_path):
            with open(self._meta_file_path, 'r') as meta_file:
                self._folds = json.load(meta_file)

    def add_fold(self, name: str, status: str) -> str:
        path = os.path.join(self._path, name)
        self._folds[name] = {'path': path, 'status': status}
        with open(self._meta_file_path, 'w') as meta_file:
            json.dump(self._folds, meta_file)
        return path

    def get_output_paths(self) -> List[str]:
        return [f['path'] for f in self._folds.values()]

    def get_folds_meta(self) -> Dict[str, Dict[str, str]]:
        return self._folds

    def get_root_dir(self) -> str:
        return self._path


class FoldedTrainer:
    def __init__(self, folds: List[str], result: FoldedTrainResult):
        self._folds = folds
        self._result = result

    def run(self, init_trainer: Callable[[FileStructManager, Dict[str, str]], Trainer]):
        for fold_num in range(len(self._folds)):
            self.train_fold(init_trainer=init_trainer, fold_id=fold_num)

    def train_fold(self, init_trainer: Callable[[FileStructManager, Dict[str, str]], Trainer], fold_id: int):
        cur_folds = self._folds.copy()
        val_fold = cur_folds.pop(fold_id)
        folds = {'train': cur_folds, 'val': val_fold}

        fold_path = self._result.add_fold(name=val_fold, status='pending')
        fsm = FileStructManager(base_dir=fold_path, is_continue=False)
        trainer = init_trainer(fsm, folds)
        trainer.train()
        self._result.add_fold(name=val_fold, status='completed')


class PipelineFoldedTrainer(FoldedTrainer, AbstractStep):
    def __init__(self, folds: StratificationResult, output_res: FoldedTrainResult):
        FoldedTrainer.__init__(self, folds=folds.get_folds(), result=output_res)
        AbstractStep.__init__(self, input_results=[folds], output_res=output_res)
