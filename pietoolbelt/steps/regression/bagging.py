import itertools
import json
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from pietoolbelt.metrics.cpu.regression import rmse


class Bagging:
    def __init__(self, predicts_path: str, main_metric: {"rmse": rmse}, workers_num: int, predicts_with_headers: bool = False,
                 predicts_names: str = 'pred_measures.csv', targets_names: str = 'ref_measures.csv'):
        self._path = predicts_path
        self._predicts_names = predicts_names
        self._targets_names = targets_names
        self._workers_num = workers_num
        self._with_headers = predicts_with_headers

        self._main_metric = {'name': list(main_metric.keys())[0], 'metric': main_metric[list(main_metric.keys())[0]]}
        self._metrics = []

    def add_metric(self, metric: callable, name: str) -> 'Bagging':
        self._metrics.append({'name': name, 'metric': metric})
        return self

    def _load_predicts(self) -> list or tuple:
        with open(os.path.join(self._path, 'meta.json'), 'r') as meta_file:
            predicts_config = json.load(meta_file)

        all_predicts = []
        headers = None
        for model in predicts_config:
            cur_fold_dir = model['path']

            with open(os.path.join(self._path, cur_fold_dir, self._predicts_names), 'r') as pred_file:
                predicts = np.loadtxt(pred_file, delimiter=',', dtype=str)
                if self._with_headers:
                    if headers is None:
                        headers = [str(p) for p in predicts[0]]
                    predicts = predicts[1:].astype(np.float32)
                else:
                    predicts = predicts.astype(np.float32)
            with open(os.path.join(self._path, cur_fold_dir, self._targets_names), 'r') as ref_file:
                targets = np.loadtxt(ref_file, delimiter=',', dtype=str)
                targets = (targets[1:] if self._with_headers else targets).astype(np.float32)

            all_predicts.append({'predicts': predicts, 'targets': targets,
                                 'model': [{'path': model['path'], 'model': model['model'], 'fold': model['fold']}]})

        if self._with_headers:
            return all_predicts, headers
        else:
            return all_predicts

    def run(self, output_path: str):
        if self._with_headers:
            all_predicts, headers = self._load_predicts()
        else:
            all_predicts = self._load_predicts()

        all_predicts_origin = [{'predicts': p['predicts'].copy(), 'targets': p['targets'].copy(), 'model': p['model']}
                               for p in all_predicts]

        all_combinations = []
        indices = list(range(len(all_predicts)))
        for cmb_len in tqdm(range(2, min(7, len(all_predicts)))):
            for cmb in itertools.combinations(indices, cmb_len):
                all_combinations.append(np.array(list(cmb), dtype=np.uint8))

        combinations_data = [[[all_predicts[i] for i in indices], indices, self._main_metric['metric']] for indices in all_combinations]
        del all_combinations

        with Pool(self._workers_num) as pool:
            metrics = list(tqdm(pool.imap(Bagging._merge_predicts_by_indices, combinations_data), total=len(combinations_data)))

        best_metric = metrics[0]['metric']
        best_predict = Bagging._merge_predicts(combinations_data[0][0])
        for idx, m in enumerate(metrics[1:]):
            if m['metric'] < best_metric:
                best_metric = m['metric']
                best_predict = Bagging._merge_predicts([all_predicts_origin[i] for i in m['indices']])

        for pred in tqdm(all_predicts_origin):
            cur_metric = Bagging._calc_metric_by_predict(pred, self._main_metric['metric'])
            if cur_metric < best_metric:
                best_metric = cur_metric
                best_predict = pred

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, 'metrics.json'), 'w') as metrics_out:
            json.dump({m['name']: Bagging._calc_metric_by_predict(best_predict, m['metric']) for m in self._metrics},
                      metrics_out, indent=4)

        with open(os.path.join(output_path, 'meta.json'), 'w') as meta_out:
            json.dump(best_predict['model'], meta_out, indent=4)

        with open(os.path.join(output_path, self._predicts_names), 'w') as pred_out, \
             open(os.path.join(output_path, self._targets_names), 'w') as ref_out:

            if self._with_headers:
                pred_out.write(','.join(headers) + '\n')
                ref_out.write(','.join(headers) + '\n')

            for pred, ref in zip(best_predict['predicts'], best_predict['targets']):
                pred_out.write(','.join([str(v) for v in pred]) + '\n')
                ref_out.write(','.join([str(v) for v in ref]) + '\n')

        return best_predict

    @staticmethod
    def _merge_predicts(predicts: []) -> dict:
        res_predict = np.median([p['predicts'] for p in predicts], axis=0)
        res_target = predicts[0]['targets']
        return {'predicts': res_predict, 'targets': res_target, 'model': [p['model'][0] for p in predicts]}

    @staticmethod
    def _merge_predicts_by_indices(data: []) -> dict:
        predicts, indices, metric = data
        cur_predict = Bagging._merge_predicts(predicts)
        return dict({'metric': Bagging._calc_metric_by_predict(cur_predict, metric), 'indices': indices},
                    **{k: v for k, v in cur_predict.items() if k not in ['predicts', 'targets']})

    @staticmethod
    def _calc_metric_by_predict(predict: {}, metric: callable) -> float:
        return metric(predict['predicts'], predict['targets'])
