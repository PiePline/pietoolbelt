import json
import os
from typing import List

from piepline.utils.fsm import FileStructManager


class FoldedTrainer:
    def __init__(self, folds: List[str]):
        self._folds = folds

    def run(self, init_trainer: callable, model_name: str, out_dir: str):
        for fold_num in range(len(self._folds)):
            self.train_fold(init_trainer=init_trainer, model_name=model_name, out_dir=out_dir, fold_num=fold_num)

    def train_fold(self, init_trainer: callable, model_name: str, out_dir: str, fold_num: int):
        cur_folds = self._folds.copy()
        val_fold = cur_folds.pop(fold_num)
        folds = {'train': cur_folds, 'val': val_fold}

        fsm = FileStructManager(base_dir=os.path.join(out_dir, model_name, val_fold), is_continue=False)
        trainer = init_trainer(fsm, folds)
        trainer.train()

        meta_info = [{'model': model_name, 'fold': val_fold, 'path': os.path.join(model_name, val_fold)}]

        self._folds.append(val_fold)

        meta_file = os.path.join(out_dir, 'meta.json')

        if os.path.exists(meta_file):
            with open(meta_file, 'r') as meta_file:
                exists_meta = json.load(meta_file)
                meta_info = exists_meta + meta_info

        with open(os.path.join(out_dir, 'meta.json'), 'w') as meta_file:
            json.dump(meta_info, meta_file, indent=4)
