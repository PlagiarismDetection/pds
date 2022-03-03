from abc import ABC


class PostProcessing(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def filter(evidences):
        pass
