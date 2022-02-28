from pds.candidate_retrieval.similarity_metric import SimilarityMetric
from sentence_transformers import SentenceTransformer
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

    def exhaustive_analysis(self, candidate_retrieval_result, ngrams=3, exact_threshold=0.95, near_threshold=0.85, paraphrase_threshold=0.7, similarity_metric=SimilarityMetric.Jaccard_2()):
        # Only Offline
        # candidate_retrieval_result: [{input_para: string, candidate_list: [{title: string, content: [source_para: string]}]}]
        evidence = []
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
                                            'cos_sim': min(1.0, cos_sim(input_sent_vector, vector).item()),
                                            'method': 'paraphrase'} for position_candidate, candidate in enumerate(source_embedding)
                                           for position_para, vector_list in enumerate(candidate['analysis_content'])
                                           for position_source, vector in enumerate(vector_list)
                                           if cos_sim(input_sent_vector, vector).item() > paraphrase_threshold]} for position_input, input_sent_vector in enumerate(input_embedding)]
            # Output: [{sent: sentence, pos: position_input, evidence: [{title: title, sent_source: sent2, sent_source_pos: position_source, cos_sim: number, method: string}]]

            # Step 4: Check if near/exact copy
            for sent_paraphrased in evidence_list:
                input_paraphrased = sent_paraphrased['sent']
                for evi in sent_paraphrased['evidence']:
                    source_paraphrased = evi['sent_source']
                    sm = self.__string_based(
                        input_paraphrased, source_paraphrased, ngrams, exact_threshold, near_threshold, similarity_metric)
                    if sm:
                        evi['method'] = sm[1]

            evidence.append(evidence_list)
        return evidence
