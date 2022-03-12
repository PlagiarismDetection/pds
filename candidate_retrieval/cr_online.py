import re
import pandas as pd
import numpy as np
import requests
from functools import reduce
from bs4 import BeautifulSoup
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

from pds.pre_processing import ViePreprocessor, EngPreprocessor
from pds.pre_processing.utils import split_para
from pds.exhaustive_analysis import SimilarityMetric
from .keyphrase_extract import KeyphraseExtract


class CROnline():
    lang = 'en'
    isPDF = False   # Text (short) or file (long)

    def __init__(self):
        pass

    @classmethod
    def chunking(cls, para_list):
        # If isPDF, Chunking each paragraph to chunk of 1 - 3 sentences.
        # Elif not isPDF, Each sentence is a chunk.
        # Use Preprocessor module to split each paragraph to list of sent.
        # Chunk has n sentences, if n/3 = 1 => semi-last and last chunk has 2 sents, else chunking to 2 - 3 sentences.
        # Combine all sentences of a chunk to 1 string, filter if chunk has less than 100 character, and add to chunklist.
        chunk_list = []

        for par in para_list:
            # Use Preprocessing to sent to split each paragraph to list of sent.
            if cls.lang == 'en':
                sent_list = EngPreprocessor.pp2sent(
                    par, replace_num=None, lowercase=False)
            else:
                sent_list = ViePreprocessor.pp2sent(
                    par, replace_num=None, lowercase=False)

            if cls.isPDF:
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

        # Filter for chunk > 80 char, and add to chunklist.
        chunk_list = [c for c in chunk_list if len(c) > 80]

        # print(len(chunk_list))
        # print([len(c) for c in chunk_list])
        # print(chunk_list)
        return chunk_list

    @classmethod
    def preprocess_chunk_list(cls, chunk_list):
        # Output: [(chunk, ppchunk)] : List of tuple of (chunk and its ppchunk) if ppchunk >= 10 words
        # Preprocessing a chunk to remove stopword and punctuation.
        pp_chunk_list = []

        for chunk in chunk_list:
            # PP each chunk to list of tokens
            pp_chunk = []
            if cls.lang == 'en':
                pp_chunk = EngPreprocessor.pp2word(
                    chunk, replace_num=None, lowercase=False)
            else:
                pp_chunk = ViePreprocessor.pp2word(
                    chunk, replace_num=None, lowercase=False)

            # Filtering word >= 4 and not contain special words.
            pp_chunk = [w for w in pp_chunk if len(w) >= 4]
            pp_chunk = [w for w in pp_chunk if w not in [
                'DATE', 'TIME', 'NUMB', 'http', 'https']]
            pp_chunk = [w for w in pp_chunk if not w.startswith(r"//")]

            pp_chunk_list.append((chunk, pp_chunk))

        # After pp2word, Filtering chunk >= 10 words If isPDF, else >=5 is ok
        pp_chunk_list = [c for c in pp_chunk_list if len(
            c[1]) >= (10 if cls.isPDF else 5)]

        return pp_chunk_list

    @classmethod
    def get_top_tf_idf_words(cls, pp_chunk_list, top_k=20):
        # instantiate the vectorizer object
        if cls.lang == 'en':
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
    def keyphrase_extract(cls, text_chunks, top_k):
        # Extract Keyphrase from list of text chunks
        kp_extractor = KeyphraseExtract(
            'english' if cls.lang == 'en' else 'vietnamese')
        kp_list = [kp_extractor.get_keyphrase(
            text, top_k=top_k) for text in text_chunks]

        # Split word in phrase of each kp => List of word only
        # Cut if kp_list is so long > 30
        kp_list = [[kp for kps in chunk for kp in kps.split(
            ' ')][:30] for chunk in kp_list]
        # print([len(c) for c in kp_list])

        # (Not) Filter all repeated word
        return kp_list

    @classmethod
    def query_formulate(cls, pp_chunk_list, top_k):
        text_chunks = [c[0] for c in pp_chunk_list]
        pp_chunks = [c[1] for c in pp_chunk_list]

        # 1st Query
        # Get first 20 word of each pp chunk
        query1_list = ["+".join(c[:top_k]) for c in pp_chunks]
        # [print(q) for q in query1_list]

        # 2nd Query
        top_list = cls.keyphrase_extract(text_chunks, top_k)
        query2_list = ["+".join(top) for top in top_list]
        # [print(q) for q in query2_list]

        return zip(text_chunks, query1_list, query2_list)

    @classmethod
    def searchBing(cls, query):
        get_header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        if cls.lang == 'en':
            url = 'https://www.bing.com/search?q=' + query + \
                "+arxiv.org+researchgate.net&go=Search&qs=ds&form=QBRE"
        elif cls.lang == 'vi':
            get_header['Accept-Language'] = 'vi-VN'
            url = 'https://www.bing.com/search?q=' + query + \
                "+researchgate.net&go=Search&qs=ds&form=QBRE"
        else:
            raise Exception(
                'SearchBing not suppported for language: ' + cls.lang)

        print(f">>> Search by Bing URL: {url}")

        s = requests.Session()
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
        # Output: [{input_para: string, candidate_list: [{title: string, content: [source_para: string], url, snippet}]}]

        result = []
        for chunk, query1, query2 in query_list:
            candidate_list = []
            candidate_list += cls.searchBing(query1)
            candidate_list += cls.searchBing(query2)

            result += [{'input_para': chunk, 'candidate_list': candidate_list}]

        return result

    @classmethod
    def download_filtering_hybrid(cls, search_results):
        for pp_res in search_results:
            candidate_list = pp_res['candidate_list']
            para_content = pp_res['input_para']
            check_duplicated = cls.check_duplicate_url(candidate_list)
            snippet_based_checking = cls.snippet_based_checking(
                check_duplicated, para_content, threshold=cls.threshold_snippet_checking)
            pp_res['candidate_list'] = snippet_based_checking

        candidate_list = cls.merge_all_result(search_results)
        return candidate_list

    @classmethod
    def merge_all_result(cls, search_results):
        candidate_list = reduce(
            lambda x, y: x+y['candidate_list'], search_results, [])
        candidate_list = cls.check_duplicate_url(candidate_list)

        return {
            'candidate_list': candidate_list
        }

    @classmethod
    def check_duplicate_url(cls, search_results):
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
            pp_title = ViePreprocessor.tokenize(
                pp_title) if cls.lang == 'vi' else EngPreprocessor.tokenize(pp_title)

            # Compaare title with alls in title_list, using Jaccard_2
            sm_title_list = [SimilarityMetric.n_gram_matching(
                pp_title, pp_found_title, 2, SimilarityMetric.Jaccard_2()) for pp_found_title in title_list]

            # print(title_list)
            # print(sm_title_list)
            max_sm = 0 if len(sm_title_list) == 0 else max(sm_title_list)
            if (url not in url_list) and max_sm <= 0.7:
                check_duplicated.append(search_res)
                url_list.append(url)
                title_list.append(pp_title)
        return check_duplicated

    @classmethod
    def snippet_based_checking(cls, search_results, suspicious_doc_string, threshold=1):
        # Check overlap on (n=3)-grams on suspicious document and candidate document
        if cls.isPDF:
            num_of_gram = 3
        else:
            num_of_gram = 2

        if cls.lang == 'vi':
            sus_preprocessed = ViePreprocessor.tokenize(
                suspicious_doc_string)
        else:
            sus_preprocessed = EngPreprocessor.tokenize(
                suspicious_doc_string)

        sus_grams = ngrams(sus_preprocessed, num_of_gram)
        sus_grams = [' '.join(grams) for grams in sus_grams]
        # print(sus_grams)

        check_snippet_based = []

        for candidate in search_results:
            if cls.lang == 'vi':
                can_preprocessed = ViePreprocessor.tokenize(
                    candidate['snippet'])
            else:
                can_preprocessed = EngPreprocessor.tokenize(
                    candidate['snippet'])

            if len(can_preprocessed) < num_of_gram:
                continue
            can_grams = ngrams(can_preprocessed, num_of_gram)
            can_grams = [' '.join(grams) for grams in can_grams]
            overlap = [value for value in sus_grams if value in can_grams]

            if len(overlap) >= threshold:
                check_snippet_based.append(candidate)

        # print(len(check_snippet_based))
        return check_snippet_based

    @classmethod
    def combine_all_step(cls, data, lang='en', isPDF=False, top_k=20, threshold_snippet_checking=1):
        # Set Language
        cls.lang = lang
        cls.isPDF = isPDF
        cls.threshold_snippet_checking = threshold_snippet_checking

        # Split data text to get important paragraph
        para_list = split_para(data, isPDF=cls.isPDF)

        # Chunking
        chunk_list = cls.chunking(para_list)
        print(f">>> Chunking to {len(chunk_list)} chunks")
        # [print(c) for c in chunk_list]

        # Preprocess chunk list
        pp_chunk_list = cls.preprocess_chunk_list(chunk_list)
        print(f"\n>>> PP Chunking to {len(pp_chunk_list)} chunks\n")
        # [print(c) for c in pp_chunk_list]

        # Searching: Query Formulate + Search control
        query_list = cls.query_formulate(pp_chunk_list, top_k)
        search_res = cls.search_control(query_list)

        res_len = [len(r['candidate_list']) for r in search_res]
        print(
            f"\n>>> Search Online found: {res_len}, total: {sum(res_len)} sources")

        # Download Filtering

        filter = cls.download_filtering_hybrid(search_res)

        filter_len = len(filter['candidate_list'])
        print(
            f"\n>>> After Filtered found: {filter_len} sources")
        [print(f) for f in filter['candidate_list']]
        filter['input_para_list'] = para_list
        return filter
