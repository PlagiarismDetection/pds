from pds.pre_processing import ViePreprocessor
from pds.pre_processing import EngPreprocessor

from pds.candidate_retrieval.similarity_metric import SimilarityMetric

from abc import ABC
import numpy as np

import re


class StringBasedTechnique(ABC):
    @staticmethod
    def __remove_dumb_sent(text):
        # cleanStr = re.sub('[.\-:]+', '', text)
        return text

    @staticmethod
    def __n_gram_matching_vie(input_sent_list, candidate_sent_list, grams_num):
        rows = len(input_sent_list)
        cols = len(candidate_sent_list)
        input_grams_list = list(map(
            lambda input_sent: ViePreprocessor.tokenize(input_sent), input_sent_list))
        candidate_grams_list = list(map(lambda candidate_sent: ViePreprocessor.tokenize(
            candidate_sent), candidate_sent_list))
        evidence_SM = np.array([])

        for input_gram in input_grams_list:
            for candidate_gram in candidate_grams_list:
                evidence_SM = np.append(evidence_SM, SimilarityMetric.n_gram_matching(
                    input_gram, candidate_gram, grams_num, SimilarityMetric.Jaccard_2()))
        return np.reshape(evidence_SM, (rows, cols))

    @staticmethod
    def __n_gram_matching_eng(input_sent_list, candidate_sent_list, grams_num):
        rows = len(input_sent_list)
        cols = len(candidate_sent_list)
        input_grams_list = list(map(
            lambda input_sent: EngPreprocessor.tokenize(input_sent), input_sent_list))
        candidate_grams_list = list(map(lambda candidate_sent: EngPreprocessor.tokenize(
            candidate_sent), candidate_sent_list))
        evidence_SM = np.array([])

        for input_gram in input_grams_list:
            for candidate_gram in candidate_grams_list:
                evidence_SM = np.append(evidence_SM, SimilarityMetric.n_gram_matching(
                    input_gram, candidate_gram, grams_num, SimilarityMetric.Jaccard_2()))
        return np.reshape(evidence_SM, (rows, cols))

    @classmethod
    def __eng_preprocessing(cls, para):
        sent_list = EngPreprocessor.pp2sent(
            para, replace_num=False, lowercase=False)
        sent_list = list(
            map(lambda sent: cls.__remove_dumb_sent(sent), sent_list))
        return list(filter(lambda sent: len(sent) != 0, sent_list))

    @classmethod
    def __vie_preprocessing(cls, para):
        sent_list = ViePreprocessor.pp2sent(
            para, replace_num=False, lowercase=False)
        sent_list = list(
            map(lambda sent: cls.__remove_dumb_sent(sent), sent_list))
        return list(filter(lambda sent: len(sent) != 0, sent_list))

    @staticmethod
    def __detect_plagiarized_list(matrix, threshold):
        value = np.amax(matrix, axis=0)
        index = np.where(matrix == value)
        result = [(index[0][i], index[1][i]) for i in range(
            len(index[0])) if matrix[index[0][i], index[1][i]] > threshold]
        return result

    @staticmethod
    def __Union(lst):
        final_list = []
        for l in lst:
            final_list = list(set().union(final_list, l))
        return final_list

    @classmethod
    def eng_string_based_technique(cls, input_paragraph, source_paragraph_list, ngrams_num, threshold):
        input_sent_list = cls.__eng_preprocessing(input_paragraph)
        list_of_source_sent_list_of_para = list(map(lambda para: {
                                                'title': para['title'], 'par': cls.__eng_preprocessing(para['par'])}, source_paragraph_list))

        num_of_para = len(source_paragraph_list)
        evidence = []
        for i in range(num_of_para):
            source_sent_list = list_of_source_sent_list_of_para[i]['par']
            evidence.append(cls.__n_gram_matching_eng(
                input_sent_list, source_sent_list, ngrams_num))

        detect_lst = list(
            map(lambda evi: cls.__detect_plagiarized_list(evi, threshold), evidence))
        input_sent_plagiarism_list = list(map(lambda evi: list(
            map(lambda i: i[0], cls.__detect_plagiarized_list(evi, threshold))), evidence))
        source_sent_plagiarism_list = list(map(lambda evi: list(
            map(lambda i: i[1], cls.__detect_plagiarized_list(evi, threshold))), evidence))
        evidence_res = []
        for i in range(num_of_para):
            title = source_paragraph_list[i]['title']
            sm_list = list(map(lambda sent_pos, pos: {'sm': evidence[i][sent_pos], 'pos': pos, 'sent': (
                list_of_source_sent_list_of_para[i]['par'])[sent_pos[1]]}, detect_lst[i], source_sent_plagiarism_list[i]))
            evidence_res.append({'title': title, 'suspicious_list': sm_list})

        sent_list_position = cls.__Union(input_sent_plagiarism_list)
        sent_list = []
        for i in sent_list_position:
            sent_list.append(input_sent_list[i])
        return {'input': {'pos': input_sent_plagiarism_list, 'sent_list': sent_list}, 'evidence': evidence_res}

    @classmethod
    def vie_string_based_technique(cls, input_paragraph, source_paragraph_list, ngrams_num, threshold):
        input_sent_list = cls.__vie_preprocessing(input_paragraph)
        list_of_source_sent_list_of_para = list(map(lambda para: {
                                                'title': para['title'], 'par': cls.__vie_preprocessing(para['par'])}, source_paragraph_list))

        num_of_para = len(source_paragraph_list)
        evidence = []
        for i in range(num_of_para):
            source_sent_list = list_of_source_sent_list_of_para[i]['par']
            evidence.append(cls.__n_gram_matching_vie(
                input_sent_list, source_sent_list, ngrams_num))

        detect_lst = list(
            map(lambda evi: cls.__detect_plagiarized_list(evi, threshold), evidence))
        input_sent_plagiarism_list = list(map(lambda evi: list(
            map(lambda i: i[0], cls.__detect_plagiarized_list(evi, threshold))), evidence))
        source_sent_plagiarism_list = list(map(lambda evi: list(
            map(lambda i: i[1], cls.__detect_plagiarized_list(evi, threshold))), evidence))
        evidence_res = []
        for i in range(num_of_para):
            title = source_paragraph_list[i]['title']
            sm_list = list(map(lambda sent_pos, pos: {'sm': evidence[i][sent_pos], 'pos': pos, 'sent': (
                list_of_source_sent_list_of_para[i]['par'])[sent_pos[1]]}, detect_lst[i], source_sent_plagiarism_list[i]))
            evidence_res.append({'title': title, 'suspicious_list': sm_list})

        sent_list_position = cls.__Union(input_sent_plagiarism_list)
        sent_list = []
        for i in sent_list_position:
            sent_list.append(input_sent_list[i])
        return {'input': {'pos': input_sent_plagiarism_list, 'sent_list': sent_list}, 'evidence': evidence_res}
