from abc import ABC
import functools

class PostProcessing(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def __post_filter(evidences):
        return list(map(lambda para_evidence: list(filter(lambda sent_evidence: sent_evidence['evidence'] != [], para_evidence)), evidences))

    @staticmethod
    def __filter_evidence_source(evidences):
        # get list title of candidate sources
        flat = [x for para in analysis_result1 for sen in para for x in sen['evidence']]
        source_list = {item['title']:item['url'] for item in flat}
        list_title = source_list.keys()
   
        evidence_source = []
        for title in list_title:
            sent_details = []
            numword_sim_source = 0
            for para in evidences:
                sent_evi_title = []
                for sent_evi in para:
                    list_evidence = list(filter(lambda evi: evi['title'] == title, sent_evi['evidence']))
                    sent_evi_title.append({
                        'evidence': list_evidence,
                        'url': source_list[title],
                        'pos': sent_evi['pos'],
                        'sent': sent_evi['sent']
                    })
                    # cal total num word plagiarism for each title
                    if len(list_evidence)>0:
                        numword_sim_source += len(sent_evi['sent'])
                sent_details.append(sent_evi_title)
            
            evidence_source.append({
                'title':title,
                'sim_score': numword_sim_source/total_word,
                'sent_details': sent_details
            })
        return evidence_source

    @classmethod
    def post_preprocessing(cls, evidences):
        # # Step 1: remove sentence with empty evidences out of evidences list
        # filtered_evidences = cls.__post_filter(evidences)

        # Initialize paramater
        numword_sim = 0
        total_word = 0
        numword_method = {
            'exact':0, 
            'near':0, 
            'paraphrase':0,
        }

        for para in evidences:
            for sent_evi in para:
                total_word += len(sent_evi['sent'])
                #check if has any plagiarism for each sentence in document
                if len(sent_evi['evidence']) > 0:
                    # cal total num word plagiarism
                    numword_sim += len(sent_evi['sent'])

                    # cal total num word for each plagiarism method
                    method_list = [evi['method'] for evi in sent_evi['evidence']]
                    for method in numword_method.keys():
                        if method in method_list:
                            numword_method[method] += len(sent_evi['sent'])
                            break

        sim_score = numword_sim/total_word
        paraphrase_score = numword_method['paraphrase']/total_word
        exact_score = numword_method['exact']/total_word
        near_score = numword_method['near']/total_word

        # Filter the evidence for each source and cal sim_score
        evidence_source = cls.__filter_evidence_source(evidences)
        
        # Output
        postprocess_result = {
            'summary':{
                'sim_score': sim_score,
                'exact_score': exact_score,
                'near_score': near_score,
                'paraphrase_score': paraphrase_score,
                'evidence_source':evidence_source,
                'optional': {} 
            },
            'sent_details':evidences
        }
        return postprocess_result
        
