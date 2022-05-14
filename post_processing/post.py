from abc import ABC
import functools


class PostProcessing(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def __filter_evidence_source(evidences, content_sources, total_word):
        # get list title of candidate sources
        flat = [x for para in evidences for sen in para for x in sen['evidence']]
        source_list = {item['title']: item['url'] for item in flat}
        list_title = source_list.keys()

        evidence_source = []
        for title in list_title:
            sent_details = []
            numword_sim_source = 0
            for para in evidences:
                sent_evi_title = []
                for sent_evi in para:
                    # filter evident for each title
                    list_evidence = list(
                        filter(lambda evi: evi['title'] == title, sent_evi['evidence']))
                    
                    # find summary_method
                    method_list = [evi['method']
                                   for evi in list_evidence]
                    
                    summary_method =''
                    for method in ['exact','near','paraphrase']:
                        if method in method_list:
                            summary_method = method
                            break

                    sent_evi_title.append({
                        'evidence': list_evidence,
                        'pos': sent_evi['pos'],
                        'sent': sent_evi['sent'],
                        'summary_method':summary_method
                    })
                    # cal total num word plagiarism for each title
                    if len(list_evidence) > 0:
                        numword_sim_source += len(sent_evi['sent'].split())
                sent_details.append(sent_evi_title)

            evidence_source.append({
                'title': title,
                'sim_score': numword_sim_source/total_word,
                'sent_details': sent_details,
                'url': source_list[title],
                'content_split': [source['analysis_content'] for source in content_sources if source['title'] == title][0]
            })
        return evidence_source

    @staticmethod
    def cal_percentage_plagiarism(evidences):
        # Initialize paramater
        numword_sim = 0
        total_word = 0
        numword_method = {
            'exact': 0,
            'near': 0,
            'paraphrase': 0,
        }

        for para in evidences['evidences']:
            for sent_evi in para:
                total_word += len(sent_evi['sent'].split())
                # check if has any plagiarism for each sentence in document
                if len(sent_evi['evidence']) > 0:
                    # cal total num word plagiarism
                    numword_sim += len(sent_evi['sent'].split())

                    # cal total num word for each plagiarism method
                    method_list = [evi['method']
                                   for evi in sent_evi['evidence']]
                    for method in numword_method.keys():
                        if method in method_list:
                            numword_method[method] += len(
                                sent_evi['sent'].split())
                            # assign summary method for sentence
                            sent_evi['summary_method'] = method
                            break

                    # summary max score for each sentence, prioritize 'exact' and 'near' method
                    near_exact_can = list(filter(lambda x: x['method'] in [
                                          'exact', 'near'], sent_evi['evidence']))
                    if len(near_exact_can) > 0:
                        sent_evi['max_score'] = max(
                            list(map(lambda x: x['sm_score'], near_exact_can)))
                    else:  # candidate_list has only paraphrase method
                        sent_evi['max_score'] = max(
                            list(map(lambda x: x['sm_score'], sent_evi['evidence'])))

        # find number of sentences plagiarism for each technique
        lst_sent_evi = [sent_evi for para in evidences['evidences'] for sent_evi in para]
        exact_sent = len(list(filter(lambda sent_evi: sent_evi['summary_method']=='exact', lst_sent_evi)))
        near_sent = len(list(filter(lambda sent_evi: sent_evi['summary_method']=='near', lst_sent_evi)))
        paraphrase_sent = len(list(filter(lambda sent_evi: sent_evi['summary_method']=='paraphrase', lst_sent_evi)))

        sim_score = numword_sim/total_word
        paraphrase_score = numword_method['paraphrase']/total_word
        exact_score = numword_method['exact']/total_word
        near_score = numword_method['near']/total_word
        return sim_score, paraphrase_score, exact_score, near_score, total_word, exact_sent, near_sent, paraphrase_sent

    @classmethod
    def post_processing(cls, evidences):
        # INPUT
        # {
        #     'evidences':evidences,
        #     'candidate_list': candidate_list_pp_sent,
        #     'input_handled': input_handled
        # }
        # OUTPUT
        # {'summary': {
        #      'sim_score': float,
        #      'paraphrase_score: float,
        #      'near_score': float,
        #      'exact_score': float,
        #      'evidence_source': [{'title': string,
        #                           'url': string
        #                           'sim_score': float,
        #                           'sent_details': [[{}]]
        #                         }]
        #      'optional': {….},
        #      },
        # 'sent_details': [[{
        #       'evidence': []
        #       'pos': int
        #       'sent': string
        #       'max_score': float
        #       'summary_method': String
        #     }]]
        # }
        filtered_evidences = evidences['evidences']

        # Calculate the percentage of plagiarism
        sim_score, paraphrase_score, exact_score, near_score, total_word = cls.cal_percentage_plagiarism(
            evidences)

        # Filter the evidence for each source and cal sim_score
        evidence_source = cls.__filter_evidence_source(
            filtered_evidences, evidences['candidate_list'], total_word)

        # Output
        postprocess_result = {
            'summary': {
                'sim_score': sim_score,
                'exact_score': exact_score,
                'near_score': near_score,
                'paraphrase_score': paraphrase_score,
                'evidence_source': evidence_source,
                'optional': {}
            },
            'sent_details': filtered_evidences,
            'input_handled': evidences['input_handled']
        }
        return postprocess_result

    @classmethod
    def post_processing_text_compare(cls, evidences):
        # INPUT
        # {
        #     'evidences':evidences,
        #     'candidate_list': candidate_list_pp_sent,
        #     'input_handled': input_handled
        # }
        # OUTPUT
        # {'summary': {
        #      'sim_score': float,
        #      'paraphrase_score: float,
        #      'near_score': float,
        #      'exact_score': float,
        #      'optional': {….},
        #      },
        # 'sent_details': [[{
        #       'evidence': []
        #       'pos': int
        #       'sent': string
        #       'max_score': float
        #       'summary_method': String
        #     }]]
        # }

        # Remove sentence with empty evidences out of evidences list
        filtered_evidences = evidences['evidences']

        # Calculate the percentage of plagiarism
        sim_score, paraphrase_score, exact_score, near_score, total_word, exact_sent, near_sent, paraphrase_sent = cls.cal_percentage_plagiarism(
            evidences)

        # Output
        postprocess_result = {
            'summary': {
                'sim_score': sim_score,
                'exact_score': exact_score,
                'near_score': near_score,
                'paraphrase_score': paraphrase_score,
                'exact_sent': exact_sent,
                'near_sent': near_sent,
                'paraphrase_sent': paraphrase_sent,
                'optional': {}
            },
            'sent_details': filtered_evidences,
            'input_handled': evidences['input_handled'],
            'source_handled': evidences['source_handled'],
        }
        return postprocess_result
