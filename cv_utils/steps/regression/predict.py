import itertools


class FoldedPredictor:
    def __init__(self, folds: [str]):
        self._folds = folds

    def run(self, init_trainer: callable):
        perms = itertools.permutations(self._folds)
        for perm in perms:
            folds = {'train': perm[:len(perm) - 1], 'val': perm[-1]}
            trainer = init_trainer(folds)
            trainer.train()
