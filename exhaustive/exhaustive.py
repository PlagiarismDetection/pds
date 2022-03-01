from pds.candidate_retrieval.similarity_metric import SimilarityMetric
from sentence_transformers.util import cos_sim
from abc import ABC


class Exhaustive(ABC):
    def __init__(self, model, preprocessor):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor

    def __preprocessing(self, para):
        sent_list = self.preprocessor.pp2sent(
            para, replace_num=False, lowercase=False)
        return list(filter(lambda sent: len(sent) != 0, sent_list))

    def __string_based(self, input_sent, source_sent, ngrams_num, exact_threshold, near_threshold, similarity_metric):
        input_grams = self.preprocessor.tokenize(input_sent)
        source_grams = self.preprocessor.tokenize(source_sent)

        similarity_score = SimilarityMetric.n_gram_matching(
            input_grams, source_grams, ngrams_num, similarity_metric)
        if similarity_score > exact_threshold:
            return (similarity_score, 'exact')
        elif similarity_score > near_threshold:
            return (similarity_score, 'near')
        return False

    @staticmethod
    def __check_paraphrasing(vector_1, vector_2, paraphrase_threshold):
        sm_score = cos_sim(vector_1, vector_2).item()
        if sm_score > paraphrase_threshold:
            return (sm_score, 'paraphrase')
        return (sm_score, None)

    def offline_exhaustive_analysis(self, candidate_retrieval_result, ngrams=3, exact_threshold=0.95, near_threshold=0.85, paraphrase_threshold=0.7, similarity_metric=SimilarityMetric.Jaccard_2()):
        evidences = []
        for para in candidate_retrieval_result:
            # Step 1: Input sentence preprocessing
            # Input paragraph pp
            input_paragraph = para['input_para']
            input_para_pp_sent = self.__preprocessing(input_paragraph)

            # Source paragraphs pp
            candidate_list = para['candidate_list']
            candidate_list_pp_sent = list(map(lambda candidate: {'title': candidate['title'], 'content': candidate['content'], 'analysis_content': [
                self.__preprocessing(source_para) for source_para in candidate['content']]}, candidate_list))

            # Step 2: Encode vector with SBERT mode
            # Loading model

            # Input: [sent: string]
            input_embedding = self.model.encode(input_para_pp_sent)
            # Output: [Vector: [number]]

            # Input: [{title: string, content: [source_para: [sent: string]]}]
            source_embedding = list(map(lambda candidate: {'title': candidate['title'], 'content': candidate['content'], 'analysis_content': list(
                map(lambda para: self.model.encode(para), candidate['analysis_content']))}, candidate_list_pp_sent))
            # Output: [{title: string, content: [source_para: [Vector: [number]]}]

            # Step 3: Check if paraphrasing
            # Input: [Vector: [number]] and [{title: string, content: [source_para: [Vector: [number]]}]
            evidence_list = [{'sent': input_para_pp_sent[position_input],
                              'pos': position_input,
                              'evidence': [{'title': candidate['title'],
                                            'sent_source': candidate_list_pp_sent[position_candidate]['analysis_content'][position_para][position_source],
                                            'sent_source_pos': position_source,
                                            'para_pos': position_para,
                                            'candidate_pos': position_candidate,
                                            'sm_score': min(1.0, self.__check_paraphrasing(input_sent_vector, vector, paraphrase_threshold)[0]),
                                            'method': self.__check_paraphrasing(input_sent_vector, vector, paraphrase_threshold)[1]}
                                           for position_candidate, candidate in enumerate(source_embedding)
                                           for position_para, vector_list in enumerate(candidate['analysis_content'])
                                           for position_source, vector in enumerate(vector_list)
                                           ]} for position_input, input_sent_vector in enumerate(input_embedding)]
            # Output: [{sent: sentence, pos: position_input, evidence: [{title: title, sent_source: sent2, sent_source_pos: position_source, cos_sim: number, method: string}]]

            # Step 4: Check if near/exact copy
            for paraphrased_evidence in evidence_list:
                for evidence in paraphrased_evidence['evidence']:
                    sm = self.__string_based(
                        paraphrased_evidence['sent'], evidence['sent_source'], ngrams, exact_threshold, near_threshold, similarity_metric)
                    if sm:
                        evidence['method'] = sm[1]

            # Step 5: Conclusion methods
            for evidence in evidence_list:
                evidence['evidence'] = list(
                    filter(lambda evi: evi['method'] != None, evidence['evidence']))

            evidences.append(evidence_list)
        return evidences

    def online_exhaustive_analysis(self, candidate_retrieval_result, ngrams=3, exact_threshold=0.95, near_threshold=0.85, paraphrase_threshold=0.8, similarity_metric=SimilarityMetric.Jaccard_2()):
        evidences = []
        # Step 1: Input sentence preprocessing for candidate source
        candidate_list=candidate_retrieval_result['candidate_list']
        # Source paragraphs pp
        candidate_list_pp_sent = list(map(lambda candidate: {'title': candidate['title'],'url':candidate['url'], 'content': candidate['content'], 'analysis_content': [
            self.__preprocessing(source_para) for source_para in candidate['content']]}, candidate_list))
        
        # Step 2: Encode vector with SBERT mode for candidate source
        # Input: [{title: string, content: [source_para: [sent: string]]}]
        source_embedding = list(map(lambda candidate: {'title': candidate['title'],'url':candidate['url'], 'content': candidate['content'], 'analysis_content': list(
            map(lambda para: self.model.encode(para), candidate['analysis_content']))}, candidate_list_pp_sent))
        # Output: [{title: string, content: [source_para: [Vector: [number]]}]    
        for input_paragraph in candidate_retrieval_result['input_para_list']: 
            # Step 1: Input sentence preprocessing          
            # Input paragraph pp
            input_para_pp_sent = self.__preprocessing(input_paragraph)

            # Step 2: Encode vector with SBERT mode
            # Input: [sent: string]
            input_embedding = self.model.encode(input_para_pp_sent)
            # Output: [Vector: [number]]

            # Step 3: Check if paraphrasing 
            # HAS THE SAME WITH OFFLINE COMPARITION IN REMAINING STEPS
            # Input: [Vector: [number]] and [{title: string, content: [source_para: [Vector: [number]]}]
            evidence_list = [{'sent': input_para_pp_sent[position_input],
                              'pos': position_input,
                              'evidence': [{'title': candidate['title'],
                                            'sent_source': candidate_list_pp_sent[position_candidate]['analysis_content'][position_para][position_source],
                                            'sent_source_pos': position_source,
                                            'para_pos': position_para,
                                            'candidate_pos': position_candidate,
                                            'sm_score': min(1.0, self.__check_paraphrasing(input_sent_vector, vector, paraphrase_threshold)[0]),
                                            'method': self.__check_paraphrasing(input_sent_vector, vector, paraphrase_threshold)[1]}
                                           for position_candidate, candidate in enumerate(source_embedding)
                                           for position_para, vector_list in enumerate(candidate['analysis_content'])
                                           for position_source, vector in enumerate(vector_list)
                                           ]} for position_input, input_sent_vector in enumerate(input_embedding)]
            # Output: [{sent: sentence, pos: position_input, evidence: [{title: title, sent_source: sent2, sent_source_pos: position_source, cos_sim: number, method: string}]]

            # Step 4: Check if near/exact copy
            for paraphrased_evidence in evidence_list:
                for evidence in paraphrased_evidence['evidence']:
                    sm = self.__string_based(
                        paraphrased_evidence['sent'], evidence['sent_source'], ngrams, exact_threshold, near_threshold, similarity_metric)
                    if sm:
                        evidence['method'] = sm[1]

            # Step 5: Conclusion methods
            for evidence in evidence_list:
                evidence['evidence'] = list(
                    filter(lambda evi: evi['method'] != None, evidence['evidence']))

            evidences.append(evidence_list)
        return evidences