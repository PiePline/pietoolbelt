from abc import ABCMeta

try:
    from apex import amp
except ImportError:
    print("Can't import NVidia apex package. Try to ")
from piepline import TrainConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer


class MixedPrecisionTrainConfig(TrainConfig, metaclass=ABCMeta):
    class _Loss(Module):
        def __init__(self, loss: Module, optimizer: Optimizer):
            super().__init__()
            self._loss = loss
            self._optimizer = optimizer

        def forward(self, *args):
            return self._loss.forward(args)

        def backward(self):
            with amp.scale_loss(self._loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()

    def __init__(self, model: Module, train_stages: [], loss: Module, optimizer: Optimizer, opt_level: str):
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        super().__init__(model, train_stages, MixedPrecisionTrainConfig._Loss(loss, optimizer), optimizer)
