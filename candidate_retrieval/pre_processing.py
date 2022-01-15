from pds.pre_processing import ViePreprocessor
from pds.pre_processing import EngPreprocessor
from abc import ABC


class Preprocessing(ABC):
    @staticmethod
    def VieCollectionProcessing(collection):
        pass

    @staticmethod
    def EngCollectionProcessing(collection):
        pass


class WordPreprocessing(Preprocessing):
    @staticmethod
    def VieCollectionProcessing(collection):
        return list(map(lambda item: ViePreprocessor.pp2word(item['Content']), collection))

    @staticmethod
    def EngCollectionProcessing(collection):
        return list(map(lambda item: EngPreprocessor.pp2word(item['Content']), collection))

    @staticmethod
    def VieFilesProcessing(files):
        return list(map(lambda item: ViePreprocessor.pp2word(item.getContent()), files))

    @staticmethod
    def EngFilesProcessing(files):
        return list(map(lambda item: EngPreprocessor.pp2word(item.getContent()), files))


class NonPreProcessing(Preprocessing):
    @staticmethod
    def VieCollectionProcessing(collection):
        return list(map(lambda item: ViePreprocessor.tokenize(item['Content']), collection))

    @staticmethod
    def EngCollectionProcessing(collection):
        return list(map(lambda item: EngPreprocessor.tokenize(item['Content']), collection))

    @staticmethod
    def VieFilesProcessing(files):
        return list(map(lambda item: ViePreprocessor.tokenize(item.getContent()), files))

    @staticmethod
    def EngFilesProcessing(files):
        return list(map(lambda item: EngPreprocessor.tokenize(item.getContent()), files))
