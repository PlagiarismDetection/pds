from pds.pre_processing import EngPreprocessor, ViePreprocessor
from pds.pre_processing.utils import remove_puntuation
from .similarity_metric import SimilarityMetric
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from abc import ABC
from pds.pre_processing.utils import split_para
import time


DEBUG = True


def log(message, param=None):
    if DEBUG:
        if param:
            print(message, param)
        else:
            print(message)

class Exhaustive(ABC):
    def __init__(self, model, preprocessor):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor

    def __preprocessing(self, para):
        sent_list = self.preprocessor.pp2sent(
            para, replace_num=False, lowercase=False)
        return list(filter(lambda sent: len(sent.split()) >= 5, sent_list))

    def __string_based(self, input_sent, source_sent, exact_threshold, near_threshold, similarity_metric):
        input_sent_rm = remove_puntuation(input_sent)
        source_sent_rm = remove_puntuation(source_sent)

        input_tokens = self.preprocessor.tokenize(input_sent_rm)
        source_tokens = self.preprocessor.tokenize(source_sent_rm)

        input_sent_rm = self.preprocessor.lowercase(input_sent_rm)
        source_sent_rm = self.preprocessor.lowercase(source_sent_rm)

        min_length = min(len(input_tokens), len(source_tokens))
        if min_length <= 5:
            n_gram = 1
        elif min_length > 5 and min_length <= 20:
            n_gram = 2
        else:
            n_gram = 3

        similarity_score = SimilarityMetric.n_gram_matching(
            input_tokens, source_tokens, n_gram, similarity_metric)
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

    def offline_exhaustive_analysis(self, candidate_retrieval_result, exact_threshold=0.95, near_threshold=0.85, paraphrase_threshold=0.7, similarity_metric=SimilarityMetric.Jaccard_2()):
        evidences = []
        input_handled = []
        summary_content_each_title = []

        for para in candidate_retrieval_result:
            # Step 1: Input sentence preprocessing
            # Input paragraph pp
            input_paragraph = para['input_para']
            input_para_pp_sent = self.__preprocessing(input_paragraph)
            input_handled.append(input_para_pp_sent)

            # Source paragraphs pp
            candidate_list = para['candidate_list']
            candidate_list_pp_sent = list(map(lambda candidate: {'title': candidate['title'], 'content': candidate['content'], 'analysis_content': [
                self.__preprocessing(source_para) for source_para in candidate['content']]}, candidate_list))

            for can in candidate_list_pp_sent:
                for source in summary_content_each_title:
                    if can['title'] == source['title']:
                        source['analysis_content'].append(
                            can['analysis_content'])
                        source['content'].append(can['content'])
                        break
                # If not found
                summary_content_each_title.append({
                    'title': can['title'],
                    'content': [can['content']],
                    'analysis_content': [can['analysis_content']]
                })

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
                                            'url':None,
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
                        paraphrased_evidence['sent'], evidence['sent_source'], exact_threshold, near_threshold, similarity_metric)
                    if sm:
                        evidence['method'] = sm[1]
                        evidence['sm_score'] = sm[0]

            # Step 5: Conclusion methods
            for evidence in evidence_list:
                evidence['evidence'] = list(
                    filter(lambda evi: evi['method'] != None, evidence['evidence']))

            evidences.append(evidence_list)
        return {
            'evidences': evidences,
            'candidate_list': summary_content_each_title,
            'input_handled': input_handled
        }

    def online_exhaustive_analysis(self, candidate_retrieval_result, exact_threshold=0.95, near_threshold=0.85, paraphrase_threshold=0.8, similarity_metric=SimilarityMetric.Jaccard_2()):
        evidences = []
        input_handled = []
        
        """ 
            Step 1: Input sentence preprocessing for candidate source 
        """
        log(">>> Step 1: Input sentence preprocessing for candidate source ")
        start_time = time.time()
        candidate_list = candidate_retrieval_result['candidate_list']
        # Source paragraphs pp
        candidate_list_pp_sent = list(map(lambda candidate: {'title': candidate['title'], 'url': candidate['url'], 'content': candidate['content'], 'analysis_content': [
            self.__preprocessing(source_para) for source_para in candidate['content']]}, candidate_list))
        log("Time execution: ", time.time() - start_time)

        """ 
            Step 2: Encode vector with SBERT model for candidate source 
        """
        log(">>> Step 2: Encode vector with SBERT model for candidate source  ")
        start_time = time.time()
        # Input: [{title: string, content: [source_para: [sent: string]]}]
        source_embedding = list(map(lambda candidate: {'title': candidate['title'], 'url': candidate['url'], 'content': candidate['content'], 'analysis_content': list(
            map(lambda para: self.model.encode(para, batch_size=64, show_progress_bar=False), candidate['analysis_content']))}, candidate_list_pp_sent))
        log("Time execution: ", time.time() - start_time)


        # Output: [{title: string, content: [source_para: [Vector: [number]]}]
        for input_paragraph in candidate_retrieval_result['input_para_list']:
            """
                Step 1: Input sentence preprocessing
            """
            log(">>> Step 1: Input sentence preprocessing ")
            start_time = time.time()
            # Input paragraph pp
            input_para_pp_sent = self.__preprocessing(input_paragraph)
            input_handled.append(input_para_pp_sent)
            log("Time execution: ", time.time() - start_time)

            """
                Step 2: Encode vector with SBERT mode
            """
            log(">>> Step 2: Encode vector with SBERT mode ")
            start_time = time.time()
            # Input: [sent: string]
            input_embedding = self.model.encode(
                input_para_pp_sent, batch_size=64, show_progress_bar=False)
            # Output: [Vector: [number]]
            log("Time execution: ", time.time() - start_time)

            """
                Step 3: Check if paraphrasing
            """
            log(">>> Step 3: Check if paraphrasing ")
            start_time = time.time()
            # HAS THE SAME WITH OFFLINE COMPARISON IN REMAINING STEPS
            # Input: [Vector: [number]] and [{title: string, content: [source_para: [Vector: [number]]}]
            evidence_list = [{'sent': input_para_pp_sent[position_input],
                              'pos': position_input,
                              'evidence': [{'title': candidate['title'],
                                            'url': candidate['url'],
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
            log("Time execution: ", time.time() - start_time)

            """
                Step 4: Check if near/exact copy
            """
            start_time = time.time()
            log(">>> Step 4: Check if near/exact copy ")
            for paraphrased_evidence in evidence_list:
                # Compare only strings has more than 2 words
                for evidence in paraphrased_evidence['evidence']:
                    sm = self.__string_based(
                        paraphrased_evidence['sent'], evidence['sent_source'], exact_threshold, near_threshold, similarity_metric)
                    if sm:
                        evidence['method'] = sm[1]
                        evidence['sm_score'] = sm[0]
            log("Time execution: ", time.time() - start_time)

            """
                Step 5: Conclusion methods
            """
            start_time = time.time()
            log(">>> Step 5: Conclusion methods ")
            for evidence in evidence_list:
                evidence['evidence'] = list(
                    filter(lambda evi: evi['method'] != None, evidence['evidence']))

            evidences.append(evidence_list)
            log("Time execution: ", time.time() - start_time)

        log(">>> End ")

        return {
            'evidences': evidences,
            'candidate_list': candidate_list_pp_sent,
            'input_handled': input_handled
        }

    def text_compare_analysis(self, input_content, source_content, is_inputPDF, is_sourcePDF, exact_threshold=0.95, near_threshold=0.85, paraphrase_threshold=0.8, similarity_metric=SimilarityMetric.Jaccard_2()):
        evidences = []
        # Step 1: Input split docs to paragraph
        # input split para
        input_split_para = split_para(input_content, is_inputPDF)
        # source split para
        source_split_para = split_para(source_content, is_sourcePDF)

        # Step 2: Sentence pre_processing
        # input sentence pp
        input_sent_pp = list(
            map(lambda para: self.__preprocessing(para), input_split_para))
        # source sentence pp
        source_sent_pp = list(
            map(lambda para: self.__preprocessing(para), source_split_para))

        # Step 3: Encode vector with SBERT mode
        # source embedding
        source_embedding = list(
            map(lambda sent_para: self.model.encode(sent_para), source_sent_pp))

        # Analysis Comparison
        for input_para_pp_sent in input_sent_pp:
            # input embedding
            input_embedding = self.model.encode(input_para_pp_sent)

            # Step 4: Check if paraphrasing
            # Input: [Vector: [number]] and [[Vector: [number]]]
            evidence_list = [{'sent': input_para_pp_sent[position_input],
                              'pos': position_input,
                              'evidence': [{
                                  'sent_source': source_sent_pp[position_para][position_source],
                                  'sent_source_pos': position_source,
                                  'para_pos': position_para,
                                  'sm_score': min(1.0, self.__check_paraphrasing(input_sent_vector, vector, paraphrase_threshold)[0]),
                                  'method': self.__check_paraphrasing(input_sent_vector, vector, paraphrase_threshold)[1]}
                for position_para, vector_list in enumerate(source_embedding)
                for position_source, vector in enumerate(vector_list)
            ]} for position_input, input_sent_vector in enumerate(input_embedding)]
            # Output: [{sent: sentence, pos: position_input, evidence: [{ sent_source: sent2, sent_source_pos: position_source, cos_sim: number, method: string}]]

            # Step 5: Check if near/exact copy
            for paraphrased_evidence in evidence_list:
                # Compare only strings has more than 2 words
                for evidence in paraphrased_evidence['evidence']:
                    sm = self.__string_based(
                        paraphrased_evidence['sent'], evidence['sent_source'], exact_threshold, near_threshold, similarity_metric)
                    if sm:
                        evidence['method'] = sm[1]
                        evidence['sm_score'] = sm[0]

            # Step 6: Conclusion methods
            for evidence in evidence_list:
                evidence['evidence'] = list(
                    filter(lambda evi: evi['method'] != None, evidence['evidence']))

            evidences.append(evidence_list)
        return {
            'evidences': evidences,
            'source_handled': source_sent_pp,
            'input_handled': input_sent_pp
        }


class VieExhaustive(Exhaustive):
    def __init__(self):
        model = SentenceTransformer('keepitreal/vietnamese-sbert')
        super().__init__(model, ViePreprocessor)


class EngExhaustive(Exhaustive):
    """
        English SBERT model Evaluation

        Dataset: Machine Translation Metrics Paraphrase Corpus
        The training set contains 5000 true paraphrase pairs and 5000 false paraphrase pairs; 
        the test set contains 1500 and 1500 pairs, respectively. The test collection from the 
        PAN 2010 plagiarism detection competition was used to generate the sentence-level PAN 
        dataset. PAN 2010 dataset consists of 41,233 text documents from Project Gutenberg in 
        which 94,202 cases of plagiarism have been inserted. The plagiarism was created either 
        by using an algorithm or by explicitly asking Turkers to paraphrase passages from the 
        original text. Only on the human created plagiarism instances were used here.

        To generate the sentence-level PAN dataset, a heuristic alignment algorithm is used to 
        find corresponding pairs of sentences within a passage pair linked by the plagiarism 
        relationship. The alignment algorithm utilized only bag-of-words overlap and length 
        ratios and no MT metrics. For negative evidence, sentences were sampled from the same 
        document and extracted sentence pairs that have at least 4 content words in common. 
        Then from both the positive and negative evidence files, training set of 10,000 sentence 
        pairs and a test set of 3,000 sentence pairs were created through random sampling.

        Link: https://github.com/wasiahmad/paraphrase_identification/tree/master/dataset/mt-metrics-paraphrase-corpus

        Result:

        model = SentenceTransformer("all-mpnet-base-v2")
        # Threshold: 0.6837999999999578 Accuracy: 0.8617127624125291
        # Threshold: 0.7007 999999999999F1: 0.8483126110124334

        model = SentenceTransformer('all-distilroberta-v1')
        # Threshold: 0.6789999999999583 Accuracy: 0.8733755414861712 
        # Threshold: 0.7001 F1: 0.8554871423397319

        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # Threshold: 0.7064999999999553 Accuracy: 0.894701766077974
        # Threshold: 0.7039999999999995 F1: 0.8915579958819493

        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        # Threshold: 0.745099999999951 Accuracy: 0.9020326557814062
        # Threshold: 0.745099999999995 F1: 0.8988608905764583

        ==> We used 'paraphrase-multilingual-MiniLM-L12-v2' model, because its performance is very good and also model size is ok. 
        To relate with our Vietnamese SBERT model, the threshold is 0.714 is nearest with this English SBERT threshold. 
    """

    def __init__(self):
        model = SentenceTransformer('all-distilroberta-v1')
        super().__init__(model, EngPreprocessor)
