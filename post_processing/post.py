from abc import ABC


class PostProcessing(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def __post_filter(evidences):
        return list(map(lambda para_evidence: list(filter(lambda sent_evidence: sent_evidence['evidence'] != [], para_evidence)), evidences))

    @classmethod
    def post_preprocessing(cls, evidences):
        # Step 1: remove sentence with empty evidences out of evidences list
        filtered_evidences = cls.__post_filter(evidences)
