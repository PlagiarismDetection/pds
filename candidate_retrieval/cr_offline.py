from pds.pre_processing import ViePreprocessor, EngPreprocessor
from pds.pre_processing.utils import remove_puntuation, split_para
from .doc2vec import SearchDoc2Vec


class CROffline():
    def __init__(self, lang='en'):
        self.lang = lang

    def retrieveCandidates(self, text, collection, isPDF=False):
        # Output: [{input_para: string, candidate_list: [{title: string, content: [source_para: string]}]}]

        model = SearchDoc2Vec.get_model(collection, self.lang)

        # Preprocess input text to word-paragraph
        input_paras = split_para(text, isPDF=isPDF)
        clean_paras = [remove_puntuation(para) for para in input_paras]

        if self.lang == 'en':
            words_paras_list = [EngPreprocessor.pp2word(
                par) for par in clean_paras]
        else:
            words_paras_list = [ViePreprocessor.pp2word(
                par) for par in clean_paras]

        # Search using Doc2vec
        search_result = SearchDoc2Vec.search(
            model, words_paras_list, collection)

        # Combine result
        result = [{'input_para': input_par, 'candidate_list': can_lst}
                  for input_par, can_lst in zip(input_paras, search_result)]

        return result
