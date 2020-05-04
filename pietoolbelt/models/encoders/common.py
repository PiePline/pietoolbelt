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
            self._layers_outputs.append(output.clone())
        return output

    def get_layers_outputs(self) -> []:
        """
        Get layers of encoder
        :return: list of encoder alyers
        """
        return self._layers_outputs

    def forward(self, *input):
        if self._collect_layers_outputs:
            self._layers_outputs = []

        return self._forward(*input)

    @abstractmethod
    def _forward(self, *input):
        """
        Internal method for inline to __forward__

        :param input: input data
        :return: result of network forward pass
        """

    @abstractmethod
    def get_layers_params(self) -> []:
        """
        Get list of layer parameters

        :return: list of layers parameters
        """