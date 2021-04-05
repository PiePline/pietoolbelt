import os
import shutil
import unittest

import torch
import numpy as np
from piepline.predict import Predictor
from piepline.utils.checkpoints_manager import CheckpointsManager
from piepline.utils.fsm import FileStructManager

from pietoolbelt.pipeline.predict.segmentation import SegmentationPredict, SegmentationPredictResult


class _ModelMock(torch.nn.Module):
    def forward(self, data):
        res = np.random.randn((1, 24, 24)).astype(np.float32)
        return torch.from_numpy(res)


class _CheckpointsManagerMock(CheckpointsManager):
    def __init__(self, fsm: 'FileStructManager'):
        pass

    def unpack(self) -> None:
        pass

    def load_model_weights(self, model: torch.nn.Module, weights_file: str = None) -> None:
        pass

    def pack(self) -> None:
        pass


class SegmentationPredictTest(unittest.TestCase):
    RESULT_DIR = 'tmp_result_dir'

    def tearDown(self):
        if os.path.exists(SegmentationPredictTest.RESULT_DIR):
            shutil.rmtree(SegmentationPredictTest.RESULT_DIR, ignore_errors=True)

    def test_init(self):
        checkpoint_manager = _CheckpointsManagerMock(fsm=FileStructManager(base_dir='tmp_dir', is_continue=True))
        model = _ModelMock()
        predictor = Predictor(model=model, checkpoints_manager=checkpoint_manager)
        SegmentationPredict(predictor=predictor, result=SegmentationPredictResult(SegmentationPredictTest.RESULT_DIR))
