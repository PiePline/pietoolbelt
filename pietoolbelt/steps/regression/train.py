import json
import os

from piepline import FileStructManager


class FoldedTrainer:
    def __init__(self, folds: [str]):
        self._folds = folds

    def run(self, init_trainer: callable, model_name: str, out_dir: str):
        meta_info = []
        for _ in range(len(self._folds)):
            val_fold = self._folds.pop(0)
            folds = {'train': self._folds, 'val': val_fold}

            fsm = FileStructManager(base_dir=os.path.join(out_dir, model_name, val_fold), is_continue=False)
            trainer = init_trainer(fsm, folds)
            trainer.train()

            meta_info.append({'model': model_name, 'fold': val_fold, 'path': os.path.join(model_name, val_fold)})

            self._folds.append(val_fold)

            with open(os.path.join(out_dir, 'meta.json'), 'w') as meta_file:
                json.dump(meta_info, meta_file, indent=4)
