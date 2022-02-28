from pds.exhaustive.exhaustive import Exhaustive
from pds.pre_processing import ViePreprocessor


class VieExhaustive(Exhaustive):
    def __init__(self, model):
        super().__init__(model, ViePreprocessor)
