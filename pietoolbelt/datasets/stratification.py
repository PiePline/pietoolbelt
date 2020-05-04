from random import randint
import numpy as np
import os

from pietoolbelt.datasets.common import BasicDataset


class DatasetStratification:
    def __init__(self, dataset: BasicDataset, calc_target_label: callable):
        self._dataset = dataset
        self._calc_label = calc_target_label
        self._progress_clbk = None

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
        indices = {}
        for i, d in enumerate(dataset):
            indices[i] = self._calc_label(d['target'])

        hist = [[] for i in range(max(indices.values()))]
        for data, idxes in indices.items():
            hist[idxes - 1].append(data)
        return np.array([len(v) for v in hist]), hist

    def stratificate_dataset(self, hist: np.ndarray, indices: {}, parts: [float]) -> []:
        res = []
        for part in parts:
            target_hist = (hist.copy() * part).astype(np.uint32)
            res.append([target_hist, self.__fill_hist(target_hist, indices)])
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
        self._dataset.set_indices(inner_indices).flush_indices(path)
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
