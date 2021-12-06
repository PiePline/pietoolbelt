from typing import Any, Callable, List

from piepline.data_producer import AbstractDataset, DataProducer

from pietoolbelt.pipeline.predict.common import AbstractPredictResult

__all__ = ['MergePredicts']


class MergePredicts:
    def __init__(self, predicts_result: List[AbstractPredictResult], dataset: AbstractDataset,
                 reduce: Callable[[List[float]], float], result: AbstractPredictResult):
        if len(predicts_result) < 2:
            raise Exception("Predicts results number must be >= 1")

        self._predicts_results = predicts_result
        self._dataset = dataset
        self._result = result
        self._reduce = reduce

        self._pick_target = lambda x: x['target'].numpy()

    def set_pick_target(self, pick: Callable[[Any], Any]) -> 'MergePredicts':
        self._pick_target = pick
        return self

    def run(self) -> AbstractPredictResult:
        dp = DataProducer(self._dataset, batch_size=1, num_workers=0).global_shuffle(False). \
            pass_indices(need_pass=True).get_loader()

        for dat in dp:
            predict = self._calc_predict(dat['data_idx'][0])
            self._result.add_predict(index=dat['data_idx'][0], predict=predict)

        return self._result

    def _calc_predict(self, index: str) -> Any:
        return self._reduce([res.get_predict(index) for res in self._predicts_results])
