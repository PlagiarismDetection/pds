import re
import pandas as pd
import numpy as np
import requests
import itertools
from bs4 import BeautifulSoup
from nltk import ngrams, word_tokenize as eng_tokenizer
from underthesea import word_tokenize as vie_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from pds.pre_processing import ViePreprocessor
from pds.pre_processing import EngPreprocessor
from pds.pre_processing.utils import split_para
from pds.candidate_retrieval.similarity_metric import SimilarityMetric

S
class CROnline():
    def __init__(self):
        pass    

    @classmethod
    def chunking(cls, data, lang='en', isPDF=False):
        # Split data text to get important paragraph
        para_list = split_para(data, isPDF=isPDF)

        # If isPDF, Chunking each paragraph to chunk of 1 - 3 sentences.
        # Elif not isPDF, Each sentence is a chunk.
        # Use Preprocessor module to split each paragraph to list of sent.
        # Chunk has n sentences, if n/3 = 1 => semi-last and last chunk has 2 sents, else chunking to 2 - 3 sentences.
        # Combine all sentences of a chunk to 1 string, filter if chunk has less than 100 character, and add to chunklist.
        chunk_list = []

        for par in para_list:
            # Use Preprocessing to sent to split each paragraph to list of sent.
            if lang == 'en':
                sent_list = EngPreprocessor.pp2sent(par)
            else:
                sent_list = ViePreprocessor.pp2sent(par)

            if isPDF:
                # Chunking each paragraph to many chunks of 3 sentences.
                chunks = [sent_list[i: i + 3]
                          for i in range(0, len(sent_list), 3)]

                # If a chunk has 4 sent => divide into 2 chunks with 2 sents.
                if len(sent_list) > 3 & len(sent_list) % 3 == 1:
                    chunks[-1] = [chunks[-2][-1]] + chunks[-1]
                    chunks[-2] = chunks[-2][:-1]

                # Combine all sentences of a chunk to 1 string
                chunk_list += [' '.join(c) for c in chunks]
            else:
                # Elif not isPDF, Each sentence is a chunk
                chunk_list += sent_list

        # Filter for chunk > 100 char, and add to chunklist.
        chunk_list = [c for c in chunk_list if len(c) > 100]

        # print(len(chunk_list))
        # print([len(c) for c in chunk_list])
        # print(chunk_list)
        return chunk_list

    @staticmethod
    def preprocess_chunk_list(chunk_list, lang='en'):
        # Preprocessing a chunk to remove stopword and punctuation.
        pp_chunk_list = []

        for chunk in chunk_list:
            # PP each chunk to list of tokens
            pp_chunk = []
            if lang == 'en':
                pp_chunk = EngPreprocessor.pp2word(
                    chunk, replace_num=True, lowercase=True)
            else:
                pp_chunk = ViePreprocessor.pp2word(
                    chunk, replace_num=True, lowercase=True)

            # Filtering word >= 4 and not contain special words.
            pp_chunk = [w for w in pp_chunk if len(w) >= 4]
            pp_chunk = [w for w in pp_chunk if w not in [
                'DATE', 'TIME', 'NUMB', 'http', 'https']]
            pp_chunk = [w for w in pp_chunk if not w.startswith(r"//")]

            pp_chunk_list.append(pp_chunk)

        # After pp2word, Filtering chunk >= 10 words.
        pp_chunk_list = [c for c in pp_chunk_list if len(c) >= 10]
        return pp_chunk_list

    @staticmethod
    def get_top_tf_idf_words(pp_chunk_list, top_k=20, lang='en'):
        # instantiate the vectorizer object
        if lang == 'en':
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        else:
            tfidf_vectorizer = TfidfVectorizer()

        # convert documents into a matrix
        chunk_list = [' '.join(c) for c in pp_chunk_list]
        features = tfidf_vectorizer.fit_transform(chunk_list)

        # retrieve the terms found in the corpora
        feature_names = np.array(tfidf_vectorizer.get_feature_names())

        # TF-IDF score matrix
        df_tfidfvect = pd.DataFrame(data=features.toarray(), index=[
                                    i for i in range(len(chunk_list))], columns=feature_names)

        sorted_nzs = np.argsort(features.data)[:-(top_k+1):-1]
        top_list = [feature_names[features.indices[sorted_nzs]]
                    for feature in features]
        return top_list

    @classmethod
    def query_formulate(cls, chunk_list, top_k=20, lang='en'):
        # Preprocess chunk list
        pp_chunk_list = cls.preprocess_chunk_list(chunk_list, lang)

        [print(c) for c in chunk_list]
        [print(c) for c in pp_chunk_list]
        # 1st Query
        # Get first 20 word of each pp chunk
        query1_list = ["+".join(c[:20]) for c in chunk_list]

        # 2nd Query
        top_list = cls.get_top_tf_idf_words(pp_chunk_list, top_k)
        query2_list = ["+".join(top) for top in top_list]

        return query1_list + query2_list

    @staticmethod
    def searchBing(query):
        get_header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        s = requests.Session()
        url = 'https://www.bing.com/search?q=' + query + \
            "+filetype%3A&go=Search&qs=ds&form=QBRE"
        print(f">>> Search by Bing URL: {url}")

        page = s.get(url, headers=get_header)
        soup = BeautifulSoup(page.text, "html.parser")

        # print(soup.find_all('li', attrs={'class': 'b_algo'}))
        output = []
        # this line may change in future based on google's web page structure
        for searchWrapper in soup.find_all('li', attrs={'class': 'b_algo'}):
            urls = searchWrapper.find_all('a')
            url = [u.get('href') for u in urls if u.get('href')][0]

            title = searchWrapper.find('a').text.strip()

            snippet = searchWrapper.find('p')
            snippet = "" if snippet == None else snippet.text

            res = {'title': title, 'url': url, 'snippet': snippet}
            output.append(res)

            print(url)

        return output

    @classmethod
    def search_control(cls, query_list):
        result = []
        for query in query_list:
            urls = cls.searchBing(query)
            result += urls

        return result

    @classmethod
    def download_filtering_hybrid(cls, search_results, lang='en'):
        for pp_res in search_results:
            candidate_list = pp_res['candidate_list']
            para_content = pp_res['input_para']
            check_duplicated = CROnline.check_duplicate_url(candidate_list)
            snippet_based_checking  = CROnline.snippet_based_checking(check_duplicated, para_content, lang='en', threshold=1)
            pp_res['candidate_list']= snippet_based_checking
        return search_results

    @staticmethod
    def check_duplicate_url(search_results, lang='en'):
        # Search result contain list of list all search results on each paragraph
        skipWebLst = ['tailieumienphi.vn', 'baovanhoa.vn', 'nslide.com', 'www.coursehero.com',
                      'towardsdatascience.com', 'medium.com']
        skipTailLst = ['model', 'aspx', 'xls', 'pptx', 'xml', 'jar', 'zip', '']

        check_duplicated = []
        url_list = []
        title_list = []


        for search_res in search_results:
            title = search_res['title']
            url = search_res['url']

            # Filter all url cannot be downloaded
            if url.split('.')[-1] in skipTailLst or url.split('/')[2] in skipWebLst:
                continue

            # PP title
            pp_title = title.lower()
            pp_title = re.sub(
                r"""[!"#$%&'()*+,\-./:;<=>?@^_`{|}~…“”–—]""", " ", pp_title)
            pp_title = vie_tokenizer(
                pp_title) if lang == 'vi' else eng_tokenizer(pp_title)

            # Compaare title with alls in title_list, using Jaccard_2
            sm_title_list = [SimilarityMetric.n_gram_matching(
                pp_title, pp_found_title, 2, SimilarityMetric.Jaccard_2()) for pp_found_title in title_list]

            # print(title_list)
            # print(sm_title_list)
            max_sm = 0 if len(sm_title_list) == 0 else max(sm_title_list)

            if (url not in url_list) and max_sm <= 0.6:
                check_duplicated.append(search_res)
                url_list.append(url)
                title_list.append(pp_title)
        return check_duplicated

    

    @staticmethod
    def snippet_based_checking(search_results, suspicious_doc_string, lang='en', threshold=1):
        # Check overlap on 3-grams on suspicious document and candidate document
        n = 3

        if lang == 'vi':
            sus_preprocessed = ViePreprocessor.tokenize(
                suspicious_doc_string)
        else:
            sus_preprocessed = EngPreprocessor.tokenize(
                suspicious_doc_string)

        sus_grams = ngrams(sus_preprocessed, n)
        sus_grams = [' '.join(grams) for grams in sus_grams]
        # print(sus_grams)

        check_snippet_based = []

        for candidate in search_results:
            if lang == 'vi':
                can_preprocessed = ViePreprocessor.tokenize(
                    candidate['snippet'])
            else:
                can_preprocessed = EngPreprocessor.tokenize(
                    candidate['snippet'])

            if len(can_preprocessed) < n:
                continue
            can_grams = ngrams(can_preprocessed, n)
            can_grams = [' '.join(grams) for grams in can_grams]
            overlap = [value for value in sus_grams if value in can_grams]

            if len(overlap) >= threshold:
                check_snippet_based.append(candidate)

        # print(len(check_snippet_based))
        return check_snippet_based

    @classmethod
    def combine_all_step(cls, data, lang='en', isPDF=False):
        chunk_list = cls.chunking(data, lang, isPDF)

        print(f">>> Chunking to {len(chunk_list)} chunks")

        query_list = cls.query_formulate(chunk_list, 20, lang)
        search_res = cls.search_control(query_list)

        print(f">>> Search Online found {len(search_res)} sources" )

        filter = cls.download_filtering_hybrid(search_res, lang)
        print(f">>> After Filtered found {len(filter)} sources")
        print(filter)

        return filter
