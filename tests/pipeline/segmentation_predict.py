import os
import shutil
import unittest

import torch
import numpy as np
from piepline.data_producer import BasicDataset, DataProducer
from piepline.predict import Predictor
from piepline.utils.checkpoints_manager import CheckpointsManager
from piepline.utils.fsm import FileStructManager

from pietoolbelt.pipeline.predict.segmentation import SegmentationPredict, SegmentationPredictResult


class _DatasetMock(BasicDataset):
    def __init__(self, num: int = 10):
        items = [np.random.randn(1, 24, 24).astype(np.float32) for _ in range(num)]
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        return {'data': item}


class _ModelMock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predicts = []

    def forward(self, data):
        res = np.random.randn(data.shape[0], 1, 24, 24).astype(np.float32)
        self.predicts.extend([np.squeeze(r) for r in res])
        return torch.from_numpy(res)


class _CheckpointsManagerMock(CheckpointsManager):
    def __init__(self, fsm: FileStructManager):
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

    def init_predictor(self) -> SegmentationPredict:
        checkpoint_manager = _CheckpointsManagerMock(fsm=FileStructManager(base_dir='tmp_dir', is_continue=True))
        model = _ModelMock()
        predictor = Predictor(model=model, checkpoints_manager=checkpoint_manager)

        try:
            result = SegmentationPredict(predictor=predictor, result=SegmentationPredictResult(SegmentationPredictTest.RESULT_DIR))
        except Exception as err:
            self.fail("SegmentationPredict initialisation failed; Error: [{}]".format(err))

        return result

    def test_init(self):
        self.init_predictor()

    def test_predict_batch_size1(self):
        predict = self.init_predictor()

        elements_num = 10
        dataset = _DatasetMock(elements_num)
        predict.run(dataset=dataset, batch_size=1, workers_num=0)

        model = predict.predictor().data_processor().model()

        dp = DataProducer(dataset, batch_size=1, num_workers=0).global_shuffle(False).pass_indices(need_pass=True).get_loader()
        real_predicts = model.predicts
        for i, data in enumerate(dp):
            cur_file = os.path.join(SegmentationPredictTest.RESULT_DIR, 'predicts', data['data_idx'][0] + '.npy')
            self.assertTrue(os.path.exists(cur_file), msg="[{}] file doesn't exists".format(cur_file))

            saved_predict = np.load(cur_file)

            self.assertTrue(np.array_equal(saved_predict, real_predicts[i]))

    def test_predict_batch_size3(self):
        predict = self.init_predictor()

        elements_num = 10
        dataset = _DatasetMock(elements_num)
        predict.run(dataset=dataset, batch_size=3, workers_num=0)

        model = predict.predictor().data_processor().model()

        dp = DataProducer(dataset, batch_size=1, num_workers=0).global_shuffle(False).pass_indices(need_pass=True).get_loader()
        real_predicts = model.predicts
        for i, data in enumerate(dp):
            cur_file = os.path.join(SegmentationPredictTest.RESULT_DIR, 'predicts', data['data_idx'][0] + '.npy')
            self.assertTrue(os.path.exists(cur_file), msg="[{}] file doesn't exists".format(cur_file))

            saved_predict = np.load(cur_file)

            self.assertTrue(np.array_equal(saved_predict, real_predicts[i]),
                            msg='saved shape: {}; real shape: {}'.format(saved_predict.shape, real_predicts[i].shape))
