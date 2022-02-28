from pds.exhaustive.exhaustive import Exhaustive
from pds.pre_processing import EngPreprocessor


class EngExhaustive(Exhaustive):
    def __init__(self, model):
        super().__init__(model, EngPreprocessor)
