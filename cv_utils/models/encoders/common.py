from abc import ABCMeta, abstractmethod

from torch.nn import Module
from torch import Tensor


class BasicEncoder(Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._collect_layers_outputs = False
        self._layers_outputs = None

    def collect_layers_outputs(self, is_need: bool):
        self._collect_layers_outputs = is_need

    def _process_layer_output(self, output: Tensor) -> Tensor:
        """
        register layer output

        Args:
            output (Tensor): output from encoder layer
        Returns:
             object of output
        """
        if self._collect_layers_outputs:
            self._layers_outputs.append(output)
        return output

    def get_layers_outputs(self) -> []:
        """
        Get layers of encoder
        :return: list of encoder alyers
        """
        return self._layers_outputs
