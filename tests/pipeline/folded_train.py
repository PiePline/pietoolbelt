import os
import shutil
import unittest
from typing import Dict, List

import torch
import numpy as np
from piepline.data_producer import BasicDataset, DataProducer
from piepline.train import Trainer
from piepline.train_config.stages import TrainStage
from piepline.train_config.train_config import BaseTrainConfig
from piepline.utils.checkpoints_manager import CheckpointsManager
from piepline.utils.fsm import FileStructManager
from torch.nn import MSELoss
from torch.optim import Adam

from pietoolbelt.pipeline.train.folded_train import FoldedTrainer, FoldedTrainResult


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._fc = torch.nn.Linear(3, 1)

    def forward(self, data):
        return self._fc(data)


class _DummyDataset(BasicDataset):
    def __init__(self):
        items = [{'data': v[:3], 'target': v[3]} for v in [np.random.randn(4).astype(np.float32)] * 100]
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        return item


class _TrainConfigMock(BaseTrainConfig):
    def __init__(self):
        data_producer = DataProducer(dataset=_DummyDataset())
        model = _DummyModel()
        super().__init__(model=model, train_stages=[TrainStage(data_producer=data_producer)], loss=MSELoss(),
                         optimizer=Adam(params=model.parameters()))


class FoldedTrainTest(unittest.TestCase):
    RESULT_DIR = 'folded_train_result'
    folds_configs = []

    def tearUp(self) -> None:
        FoldedTrainTest.folds_configs = []

    def tearDown(self) -> None:
        if os.path.exists(FoldedTrainTest.RESULT_DIR):
            shutil.rmtree(FoldedTrainTest.RESULT_DIR, ignore_errors=True)
        FoldedTrainTest.folds_configs = []

    def test_init(self):
        try:
            FoldedTrainer(folds=['fold_1', 'fold_2'], result=FoldedTrainResult(path=self.RESULT_DIR))
        except Exception as err:
            self.fail("Can't instantiate FoldedTrain. Error: [{}]".format(err))

    def _init_trainer(self, fsm: FileStructManager, folds: Dict[str, str]) -> Trainer:
        trainer = Trainer(train_config=_TrainConfigMock(), fsm=fsm, device='cpu').set_epoch_num(2)
        CheckpointsManager(fsm=fsm).subscribe2trainer(trainer)
        self.folds_configs.append(folds.copy())
        return trainer

    def _init_folded_trainer(self, folds_names: List[str]):
        return FoldedTrainer(folds=folds_names, result=FoldedTrainResult(path=self.RESULT_DIR))

    def test_train(self):
        folds_names = ['fold_1', 'fold_2']
        trainer = self._init_folded_trainer(folds_names)

        try:
            trainer.run(init_trainer=self._init_trainer)
        except Exception as err:
            self.fail("FoldedTrain run failed. Error: [{}]".format(err))

        self.assertTrue(os.path.exists(self.RESULT_DIR))
        self.assertTrue(os.path.exists(os.path.join(self.RESULT_DIR, 'meta.json')))

        self.assertEqual(self.folds_configs, [{'train': ['fold_2'], 'val': 'fold_1'}, {'train': ['fold_1'], 'val': 'fold_2'}])

    def test_train_fold(self):
        folds_names = ['fold_1', 'fold_2']
        trainer = self._init_folded_trainer(folds_names)

        try:
            trainer.train_fold(init_trainer=self._init_trainer, fold_id=1)
        except Exception as err:
            self.fail("FoldedTrain run failed. Error: [{}]".format(err))

        self.assertTrue(os.path.exists(self.RESULT_DIR))
        self.assertTrue(os.path.exists(os.path.join(self.RESULT_DIR, 'meta.json')))

        self.assertEqual(self.folds_configs, [{'train': ['fold_1'], 'val': 'fold_2'}])

    def test_folds_combination(self):
        folds_names = ['fold_1', 'fold_2', 'fold_3']
        trainer = self._init_folded_trainer(folds_names)

        try:
            trainer.run(init_trainer=self._init_trainer)
        except Exception as err:
            self.fail("FoldedTrain run failed. Error: [{}]".format(err))

        self.assertTrue(os.path.exists(self.RESULT_DIR))
        self.assertTrue(os.path.exists(os.path.join(self.RESULT_DIR, 'meta.json')))

        self.assertEqual(self.folds_configs, [{'train': ['fold_2', 'fold_3'], 'val': 'fold_1'},
                                              {'train': ['fold_1', 'fold_3'], 'val': 'fold_2'},
                                              {'train': ['fold_1', 'fold_2'], 'val': 'fold_3'}])
