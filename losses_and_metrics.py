def split_masks_by_classes(pred: Tensor, target: Tensor):
    preds = torch.split(pred, 1, dim=1)
    targets = torch.split(target, 1, dim=1)

    return list(zip(preds, targets))


def dice(pred: torch.Tensor, target: torch.Tensor, need_numpy: bool, eps: float = 1e-7):
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = torch.sum(iflat * tflat)

    res = (2. * intersection + eps) / (torch.sum(iflat) + torch.sum(tflat) + eps)
    if need_numpy:
        res = float(res)

    return res


class Activation(Module):
    def __init__(self, activation: str = None):
        super().__init__()
        self._activation = lambda x: x
        if activation is not None:
            if activation == 'sigmoid':
                self._activation = torch.nn.Sigmoid()
            else:
                raise NotImplementedError("Activation '{}' not implemented".format(activation))

    def forward(self, input):
        return self._activation(input)


class ModelWithActivation(Module):
    def __init__(self, base_model: Module, activation: str):
        super().__init__()
        self._base_model = base_model
        self._activation = Activation(activation)

    def forward(self, data):
        res = self._base_model(data)
        return self._activation(res)


class DiceMetric(AbstractMetric):
    def __init__(self, activation: str = None):
        super().__init__('dice')
        self._activation = Activation(activation)

    def calc(self, output: torch.Tensor, target: torch.Tensor) -> np.ndarray or float:
        return dice(self._activation(output), target, True)


eps = 1e-6


def jaccard(preds: torch.Tensor, trues: torch.Tensor):
    preds_inner = preds.cpu().data.numpy().copy()
    trues_inner = trues.cpu().data.numpy().copy()

    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.size // preds_inner.shape[0]))
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.size // trues_inner.shape[0]))
    intersection = (preds_inner * trues_inner).sum(1)
    scores = (intersection + eps) / ((preds_inner + trues_inner).sum(1) - intersection + eps)

    return scores


class DiceLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-7, activation: str = None):
        super().__init__()
        self._activation = Activation(activation)
        self._eps = eps

    def forward(self, output, target):
        return 1 - dice(self._activation(output), target, False, eps=self._eps)


class JaccardMetric(AbstractMetric):
    def __init__(self, activation: str = None):
        super().__init__('jaccard')
        self._activation = Activation(activation)

    def calc(self, output: torch.Tensor, target: torch.Tensor) -> np.ndarray or float:
        return jaccard(self._activation(output), target)

    @staticmethod
    def min_val() -> float:
        return 0

    @staticmethod
    def max_val() -> float:
        return 1


class KoeffMetric(AbstractMetric):
    def __init__(self, name: str, param: torch.nn.Parameter):
        super().__init__(name)
        self.param = param

    def calc(self, output: torch.Tensor, target: torch.Tensor) -> np.ndarray or float:
        return self.param.data


class SegmentationMetricsProcessor(MetricsProcessor):
    def __init__(self, stage_name: str):
        super().__init__()
        self.add_metrics_group(MetricsGroup(stage_name).add(JaccardMetric(activation='sigmoid')).add(DiceMetric(activation='sigmoid')))


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
            self._coeffs[i] = torch.nn.Parameter(torch.Tensor(np.array([c], dtype=np.float32)), requires_grad=True)

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


class MulticlassSegmentationLoss(Module):
    def __init__(self, base_loss):
        super().__init__()
        self._base_loss = base_loss

    def forward(self, output, target):
        overall_res = []
        for p, t in split_masks_by_classes(output, target):
            overall_res.append(self._base_loss(p, t))
        return sum(overall_res)


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_w: float, dice_w: float):
        super().__init__()

        bce = torch.nn.BCEWithLogitsLoss()
        dice = DiceLoss(eps=1e-7)

        self._loss = MulticlassSegmentationLoss(ComposedLoss([bce, dice], [bce_w, dice_w]))

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return self._loss(output, target)
