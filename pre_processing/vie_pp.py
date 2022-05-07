# library for regular expression operations
import re
# library for standardlize VNM, download from https://gist.github.com/nguyenvanhieuvn/72ccf3ddf7d179b281fdae6c0b84942b
from .utils import *
# library for VNM word tokenize
from underthesea import word_tokenize, sent_tokenize

# Import the Vietnamese stopwords file, download from: https://github.com/stopwords/vietnamese-stopwords
f = open('pds/pre_processing/stopwords/vietnamese-stopwords.txt', encoding="utf8")
vnm_stopwords = f.read().splitlines()
f.close()

# Get punctuations sring from NLTK
punctuations = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~…“”–"""


class ViePreprocessor():
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

        text = cls.standardize_unicode(text)
        text = cls.standardize_marks(text)

        if lowercase:
            text = text.lower()

        sent_list = sent_tokenize(text)

        return sent_list

    @classmethod
    def pp2word(cls, input, replace_num=True, lowercase=True):
        text = input

        if replace_num == None:
            # Remove number
            text = cls.replace_number(text, remove=True)
        elif replace_num:
            # Replace numb
            text = cls.replace_number(text, remove=False)

        text = cls.standardize_unicode(text)
        text = cls.standardize_marks(text)

        tokens = cls.tokenize(text)
        tokens_clean = cls.rm_stopword_punct(tokens)

        if lowercase:
            tokens_clean = cls.lowercase(tokens_clean)

        return tokens_clean

    @staticmethod
    def tokenize(text):
        return word_tokenize(text)

    @classmethod
    def replace_number(cls, text, remove=False):
        newtext = text

        # remove date time ?
        newtext = re.sub(r'\d+[/-]\d+([/-]\d+)*',
                         '' if remove else ' DATE', newtext)
        newtext = re.sub(r'\d+[:]\d+([:]\d+)*',
                         '' if remove else ' TIME', newtext)

        # remove currency ?
        # newtext = re.sub(r'\d+([.,]\d+)*$', ' dollar', newtext)
        # newtext = re.sub(r'$\d+([.,]\d+)*', ' dollar', newtext)

        # remove simple int number, float number may be following space or "(" like "(12.122.122)"
        newtext = re.sub(r'-?\d+([.,]\d+)*',
                         '' if remove else ' NUMB', newtext)
        return newtext

    @classmethod
    def standardize_unicode(cls, text):
        std_uni_text = convert_unicode(text)
        return std_uni_text

    @classmethod
    def standardize_marks(cls, text):
        std_marks_text = chuan_hoa_dau_cau_tieng_viet(text)
        return std_marks_text

    @classmethod
    def lowercase(cls, tokens):
        tokens_clean = [word.lower() for word in tokens]
        return tokens_clean

    @classmethod
    def rm_stopword_punct(cls, tokens):
        tokens_clean = []

        for word in tokens:                         # Go through every word in your tokens list
            word = word.lower()                     # lowercase

            if '..' in word:
                continue
            if word[:2] == '. ':
                word = word[2:]

            # word = re.sub(r'(\[])(\]\s)?(\[\s\]\s?)*', '', word) # Replace '] [ ] [ ]' => ''
            word = re.sub(r'[\[\]+\'\"]', '', word) # Replace '] [] []' => ''
            word = re.sub(r'\s*$', '', word)        # Remove 'kmeans  ' => 'kmeans'

            if word == '': continue
            
            if (word not in vnm_stopwords and       # remove stopwords
                    word not in punctuations):      # remove punctuation
                tokens_clean.append(word)

        return tokens_clean
