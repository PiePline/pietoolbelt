import torch
from torch.nn import Module


class ComposedLoss(Module):
    def __init__(self, losses: [], coeffs: [] = None):
        super().__init__()
        self._losses = losses
        if coeffs is None:
            self._coeffs = [1 / len(losses) for _ in losses]
        else:
            if len(coeffs) != len(losses):
                raise Exception("Number of coefficients ({}) doesn't equal to number of losses ({})".format(len(coeffs), len(losses)))
            self._coeffs = coeffs

        self._apply_weight = lambda l, c: c * l

    @staticmethod
    def _multiply_coeff_with_exp(loss_val, weight):
        return torch.exp(-weight) * loss_val + weight

    def enable_learn_coeffs(self, strategy: str = 'exp'):
        for i, c in enumerate(self._coeffs):
            self._coeffs[i] = torch.nn.Parameter(torch.tensor([c], dtype=torch.float32), requires_grad=True)

        if strategy == 'exp':
            self._apply_weight = self._multiply_coeff_with_exp
        else:
            raise Exception("Strategy '{}' doesn't have implementation. ComposedLoss have only 'exp'".format(strategy))

    def get_coeffs(self) -> []:
        return self._coeffs

    def forward(self, *args, **kwargs):
        res = 0

        for l, c in zip(self._losses, self._coeffs):
            res += self._apply_weight(l(*args, **kwargs), c)

        return res
