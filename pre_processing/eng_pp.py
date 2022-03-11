import re

from nltk import word_tokenize, sent_tokenize
# module for stop words that come with NLTK
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer         # module for stemming
from nltk.stem import WordNetLemmatizer     # module for lemmatization

# Get punctuations sring from NLTK
punctuations = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~…“”–"""


class EngPreprocessor():
    def __init__(self):
        pass

    @classmethod
    def pp2sent(cls, input, replace_num=True, lowercase=True):
        text = input
        
        if replace_num == None:
            # Remove number
            text = cls.replace_number(text, remove=True)
        elif replace_num:
            # Replace numb
            text = cls.replace_number(text, remove=False)
            
        if lowercase:
            text = cls.lowercase(text)
        
        sent_list = sent_tokenize(text)

        return sent_list

    @classmethod
    def pp2word(cls, input, replace_num=True, lowercase=True, stem=False, lemmatize=False):
        text = input

        if replace_num == None:
            # Remove number
            text = cls.replace_number(text, remove=True)
        elif replace_num:
            # Replace numb
            text = cls.replace_number(text, remove=False)

        if lowercase:
            text = cls.lowercase(text)

        tokens = cls.tokenize(text)
        tokens_clean = cls.rm_stopword_punct(tokens)

        if stem:
            tokens_clean = cls.stemming(tokens_clean)
        elif lemmatize:
            tokens_clean = cls.lemmatize(tokens_clean)

        return tokens_clean

    @classmethod
    def replace_number(cls, text, remove=False):
        newtext = text

        # remove date time ?
        newtext = re.sub(r'\d+[/-]\d+([/-]\d+)*', '' if remove else ' DATE', newtext)
        newtext = re.sub(r'\d+[:]\d+([:]\d+)*', '' if remove else ' TIME', newtext)

        # remove currency ?
        # newtext = re.sub(r'\d+([.,]\d+)*$', ' dollar', newtext)
        # newtext = re.sub(r'$\d+([.,]\d+)*', ' dollar', newtext)

        # remove simple int number, float number may be following space or "(" like "(12.122.122)"
        newtext = re.sub(r'-?\d+([.,]\d+)*', '' if remove else ' NUMB', newtext)
        return newtext

    @classmethod
    def lowercase(cls, text):
        text1 = text
        return text1.lower()

    @staticmethod
    def tokenize(text):
        return word_tokenize(text)

    @classmethod
    def rm_stopword_punct(cls, tokens):
        stopwords_english = stopwords.words('english')
        tokens_clean = []

        for word in tokens:                         # Go through every word in your tokens list
            if (word not in stopwords_english and   # remove stopwords
                    word not in punctuations):      # remove punctuation
                tokens_clean.append(word)
        return tokens_clean

    @classmethod
    def stemming(cls, tokens):
        # Instantiate stemmer class
        stemmer = PorterStemmer()

        # Create an empty list to store the stems
        tokens_stem = []

        for word in tokens:
            stem_word = stemmer.stem(word)  # stem word
            tokens_stem.append(stem_word)   # append to the list
        return tokens_stem

    @classmethod
    def lemmatize(cls, tokens):
        # Instantiate lemmatizer class
        lemmatizer = WordNetLemmatizer()

        # Create an empty list to store the stems
        tokens_lemma = []

        for word in tokens:
            lemma_word = lemmatizer.lemmatize(word)  # lemmatize word
            tokens_lemma.append(lemma_word)         # append to the list
        return tokens_lemma
