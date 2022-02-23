from .string_based import StringBasedTechnique
from pds.pre_processing import EngPreprocessor
from pds.candidate_retrieval.similarity_metric import SimilarityMetric


class EngExhaustive():
    def __init__(self):
        pass

    @staticmethod
    def __preprocessing(para):
        sent_list = EngPreprocessor.pp2sent(
            para, replace_num=False, lowercase=False)
        return list(filter(lambda sent: len(sent) != 0, sent_list))

    @staticmethod
    def __encode():
        pass

    @staticmethod
    def __cosine_similarity():
        pass

    @staticmethod
    def __string_based():
        pass

    def exhaustive_analysis(self, candidate_retrieval_result, ngrams=3, ngram_threshold=0.9, paraphrase_threshold=0.7):
        # Only Offline
        # candidate_retrieval_result: [{input_para: string, candidate_list: [{title: string, content: [source_para: string]}]}]
        evidence = []
        for para in candidate_retrieval_result:
            # Step 1: Input sentence preprocessing
            # Input paragraph pp
            input_paragraph = para.input_para
            input_para_pp_sent = self.__preprocessing(input_paragraph)

            # Source paragraphs pp
            candidate_list = candidate_retrieval_result.candidate_list
            candidate_list_pp_sent = list(map(lambda candidate: {'title': candidate.title, 'content': [
                                          self.__preprocessing(source_para) for source_para in candidate.content]}, candidate_list))

            # Step 2: Encode vector with SBERT mode
            # Input: [sent: string]
            # Output: [Vector: [number]]

            for candidate in candidate_list_pp_sent:
                # Input: {title: string, content: [source_para: [sent: string]]}
                break
                # Output: {title: string, content: [source_para: [Vector: [number]]}

            # Step 3: Check if paraphrasing
            # Input: [Vector: [number]] and [{title: string, content: [source_para: [Vector: [number]]}]
            #   paraphrase_evidence_list = []
            #   for sentence in encoded_sent_input:
            #       paraphrase_evidence_unit = []
            #       position_input = encoded_sent_input.index(sentence)
            #       for candidate in candidate_list_pp_encoded_sent:
            #           get title
            #           paraphrase_evidence_unit.title = []
            #           obtain the list of vectors: [Vector: [number]]
            #           for vector in above list:
            #               calculate the cosine similarity
            #               position_source = list.index(vector)
            #               if cos_sim > threshold: paraphrase_evidence_unit.append({title: title, sent_source: sent2, sent_source_pos: position_source, cos_sim: number. method: string})
            #       paraphrase_evidence_list.append({sent: sentence, pos: position_input, evidence: paraphrase_evidence_unit})

            # Output: [{sent: sentence, pos: position_input, evidence: [{title: title, sent_source: sent2, sent_source_pos: position_source, cos_sim: number, method: string}]]

            # Step 4: Check if near/exact copy
            # for each sent_pair:
            # if StringBasedTechnique.eng_string_based_technique(sent1, sent2, ngrams_num, ngram_threshold):
            #       update method

            # Output: [{sent: sentence, pos: position_input, evidence: [{title: title, sent_source: sent2, sent_source_pos: position_source, cos_sim: number, method: string}]]
