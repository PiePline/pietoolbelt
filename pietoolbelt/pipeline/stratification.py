import json
from multiprocessing import Pool
from random import randint
from typing import List

import numpy as np
import os

from tqdm import tqdm

from pietoolbelt.datasets.common import BasicDataset
from pietoolbelt.pipeline.abstract_step import AbstractStep, DatasetInPipeline, AbstractStepDirResult


class StratificationResult(AbstractStepDirResult):
    def __init__(self, path: str, allow_exist=False):
        super().__init__(path, allow_exist=allow_exist)
        self._meta_file = os.path.join(path, 'meta.json')

        if os.path.exists(self._meta_file):
            with open(self._meta_file, 'r') as meta_file:
                self._meta = json.load(meta_file)
        else:
            self._meta = dict()

        self._name2file = lambda name: name + '.npy' if len(name) < 4 or name[-4:] != '.npy' else name

    def add_indices(self, indices: List[np.uint], name: str, dataset: BasicDataset):
        dataset.set_indices(indices).flush_indices(os.path.join(self._path, self._name2file(name)))

        self._meta[name] = {'indices_num': len(indices)}

        with open(self._meta_file, 'w') as meta_file:
            json.dump(self._meta, meta_file)

    def get_indices(self, name: str) -> List[np.ndarray]:
        file_path = os.path.join(self._path, self._name2file(name))
        if not os.path.exists(file_path):
            raise RuntimeError('Indices file doesnt exists [{}]'.format(file_path))

        return np.load(file_path)

    def get_output_paths(self) -> List[str]:
        return [self._path]


class DatasetStratification:
    def __init__(self, dataset: BasicDataset, calc_target_label: callable, result: StratificationResult, workers_num: int = 0):
        self._dataset = dataset
        self._calc_label = calc_target_label
        self._progress_clbk = None
        self._workers_num = workers_num
        self._result = result

    @staticmethod
    def __fill_hist(target_hist: [], indices: {}):
        def pick(d):
            idx = randint(0, len(indices[d]) - 1)
            res = indices[d][idx]
            del indices[d][idx]
            return res

        res = {}
        for idx, d in enumerate(target_hist):
            idxes = []
            for _ in range(d):
                idxes.append(pick(idx))
            res[idx] = idxes
        return res

    def calc_hist(self, dataset: BasicDataset):
        labels = []

        if self._workers_num > 1:
            with Pool(self._workers_num) as pool, tqdm(total=len(dataset)) as pbar:
                for label in pool.imap(self._calc_label, (d['target'] for d in dataset), chunksize=self._workers_num * 10):
                    labels.append(label)
                    pbar.update()
        else:
            for d in tqdm(dataset, total=len(dataset)):
                labels.append(self._calc_label(d['target']))

        hist = [[] for _ in range(max(labels))]
        for i, idxes in enumerate(labels):
            hist[idxes - 1].append(i)
        return np.array([len(v) for v in hist]), hist

    def stratificate_dataset(self, hist: np.ndarray, indices: list, parts: [float]) -> []:
        res = []
        for part in parts[:len(parts) - 1]:
            target_hist = (hist.copy() * part).astype(np.uint32)
            res.append([target_hist, self.__fill_hist(target_hist, indices)])
        res.append([np.array([len(i) for i in indices]).astype(np.uint32), {i: v for i, v in enumerate(indices)}])
        return res

    @staticmethod
    def check_indices_for_intersection(indices: []):
        for i in range(len(indices)):
            for index in indices[i]:
                for other_indices in indices[i + 1:]:
                    if index in other_indices:
                        raise Exception('Indices intersects')

    def balance_classes(self, hist: np.ndarray, indices: {}) -> tuple:
        target_hist = hist.copy()
        target_hist[np.argmax(target_hist)] = np.sum(target_hist[target_hist != target_hist.max()])
        return target_hist, self.__fill_hist(target_hist, indices)

    def _flush_indices(self, indices: [], part_indices: [], path: str):
        inner_indices = [part_indices[it] for bin in indices[1].values() for it in bin]
        self._result.add_indices(inner_indices, path, self._dataset)
        return inner_indices

    def run(self, parts: {str: float}, out_dir_path: str) -> None:
        if not os.path.exists(out_dir_path):
            raise Exception("Output dir doesn't exist '{}'".format(out_dir_path))

        hist, indices = self.calc_hist(self._dataset)

        parts = [[path, part] for path, part in parts.items()]
        pathes = [p[0] for p in parts]
        parts = [p[1] for p in parts]
        stratificated_indices = self.stratificate_dataset(hist, indices, parts)
        part_indices = {i: i for i in range(len(self._dataset))}

        indices_to_check = []
        for i, cur_indices in enumerate(stratificated_indices):
            indices_to_check.append(self._flush_indices(cur_indices, part_indices, os.path.join(out_dir_path, pathes[i])))

        self.check_indices_for_intersection(indices_to_check)


class PipelineDatasetStratification(DatasetStratification, AbstractStep):
    def __init__(self, dataset: DatasetInPipeline, calc_target_label: callable, result: StratificationResult, workers_num: int = 1):
        DatasetStratification.__init__(self, dataset, calc_target_label, result=result, workers_num=workers_num)
        AbstractStep.__init__(self, input_results=[dataset], output_res=result)
