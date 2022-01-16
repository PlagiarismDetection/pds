from nltk import download
from .eng_pp import *
from .vie_pp import *

# download the English stopwords from NLTK
download('stopwords')
# download the pre-trained Punkt tokenizer for English, using for PorterStemmer module
download('punkt')
# download wornet for English, using for WordNetLemmatizer module
download('wordnet')


# Example Demo for Using
# Please read the documentation before using!!!
# Note: All boolean parameter for each function in this demo is also its default.
if __name__ == '__main__':
    data = """ Long short-term memory(LSTM) is an artificial recurrent neural network(RNN) architecture used in the field of deep learning.  Unlike standard feedforward neural networks, LSTM has a feedback connection.  It can process not only single data points but also entire sequences of data.  LSTM is applicable to tasks such as writing and speech recognition.
                
            You can use an RNN using LSTM units in supervised fashion, using an optimization algorithm, like gradient descent, combined with backpropagation through time to calculate the gradients needed during the optimization process, so that you can change each weight of the LSTM network in proportion to the derivative of the error (at the output layer of the LSTM network) with respect to corresponding weight.
            """

    from pds.pre_processing.utils import split_para
    from pds.pre_processing import ViePreprocessor
    from pds.pre_processing import EngPreprocessor

    # Preprocess to paragraphs
    para_list = split_para(data, isPDF=False)

    # Preprocess to sentences
    sent_list = EngPreprocessor.pp2sent(
        data, replace_num=True, lowercase=True)

    sent_list = ViePreprocessor.pp2sent(
        data, replace_num=True, lowercase=True)

    # Preprocess to word tokens
    # Always including tokenizing and removing stop words & punctuations
    word_list = EngPreprocessor.pp2word(
        data, replace_num=True, lowercase=True, stem=False, lemmatize=False)

    word_list = ViePreprocessor.pp2word(
        data, replace_num=True, lowercase=True)

    # Tokenize only!
    word_list = EngPreprocessor.tokenize(data)
    word_list = ViePreprocessor.tokenize(data)
