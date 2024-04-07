from typing import Iterable, Union

import torch.nn as nn


class FreezeParameters:
    def __init__(self, modules: Union[Iterable[nn.Module], nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases this can speed up computation because gradients aren't calculated for these modules.
        Example usage:
            with FreezeParameters([module]):
                output_tensor = module(input_tensor)

        :param modules: An iterable of modules or a single module. Used to call .parameters() to freeze gradients.
        """
        # Ensure modules is a list for consistency
        if not isinstance(modules, Iterable):
            modules = [modules]
        self.modules = modules
        self.original_requires_grad = []  # Keep track of original requires_grad states

    def __enter__(self):
        # Freeze parameters
        for module in self.modules:
            for param in module.parameters():
                self.original_requires_grad.append(param.requires_grad)
                param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Unfreeze parameters
        i = 0  # Index to track position in original_requires_grad
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = self.original_requires_grad[i]
                i += 1
        self.original_requires_grad.clear()
