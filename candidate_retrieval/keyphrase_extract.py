from rake_nltk import Rake

from pds.pre_processing import ViePreprocessor, EngPreprocessor
from pds.exhaustive_analysis.similarity_metric import SimilarityMetric

# Import the Vietnamese stopwords file, download from: https://github.com/stopwords/vietnamese-stopwords
vn_stopwords = open('pds/pre_processing/stopwords/vietnamese-stopwords.txt',
                    encoding="utf8").read().splitlines()

# English stopwords list from SMART (Salton,1971).  Available at ftp://ftp.cs.cornell.edu/pub/smart/english.stop
en_stopwords = open('pds/pre_processing/stopwords/english-stopwords.txt',
                    encoding="utf8").read().splitlines()

# Get punctuations sring from NLTK
punctuations = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~…“”–"""


class KeyphraseExtract():
    def __init__(self, lang='english'):
        if lang == 'english':
            self.stopwords = set(en_stopwords)
            self.tokenizer = EngPreprocessor.tokenize
        elif lang == 'vietnamese':
            self.stopwords = set(vn_stopwords)
            self.tokenizer = ViePreprocessor.tokenize
        else:
            raise Exception(
                'Keyphrase Extraction not suppported for language: ' + lang)

        self.lang = lang

    def extract_keyphrase(self, text):
        # Output list of pair (ranks, phrases)
        # ranks is pair (score, phrase)
        # phrases is list of phrase

        # Using max=4 min=1 words each phrase
        rake = Rake(self.stopwords, set(punctuations), language=self.lang, max_length=4, min_length=1,
                    include_repeated_phrases=False, word_tokenizer=self.tokenizer)

        # Extract keyphrase
        rake.extract_keywords_from_text(text)

        ranks = rake.get_ranked_phrases_with_scores()
        phrases = rake.get_ranked_phrases()
        return ranks, phrases

    def filter_keyphrase(self, ranks, phrases):
        # Remove repeated phrases with Jaccard_2 Similarity Score >= 0.65
        filter_ranks = []

        for i in range(len(phrases)):
            for j in range(i):
                sim = SimilarityMetric.n_gram_matching(
                    phrases[i], phrases[j], 2, SimilarityMetric.Jaccard_2())

                if sim >= 0.65:
                    break
            else:
                filter_ranks.append(ranks[i])

        return filter_ranks

    def get_keyphrase(self, text, top_k=20):
        # Output list of keyphrase

        # Using RAKE for keyphrase extraction
        ranks, phrases = self.extract_keyphrase(text)

        # PostProcess: Filter repeated phrases with Jaccard_2 Similarity Score >= 0.65
        filter_ranks = self.filter_keyphrase(ranks, phrases)

        # Get ranks with score=1.0 (maybe phrase 1 word)
        # Since phrases with 1 word are often underated
        # Then sort list by length
        underated_ranks = [r for r in filter_ranks if r[0] == 1.0]
        underated_ranks = sorted(underated_ranks, key=lambda x: len(x[1]))

        # Combine top_k keyphrase = top_k/2 [filter_ranks] + top_k/2 [underated_ranks]
        k_2 = int(top_k/2)
        top_k_keyphrase = [r[1] for r in filter_ranks][:k_2] + [r[1]
                                                                for r in underated_ranks][:k_2]
        return top_k_keyphrase
