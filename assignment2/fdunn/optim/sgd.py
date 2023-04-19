"""
sgd optimizer
"""
import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, model, lr=0.0):
        self.model = model
        self.lr = lr

    def step(self):
        """Performs a single optimization step.
        """
        pass