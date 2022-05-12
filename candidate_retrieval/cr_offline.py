# -*- coding: utf-8 -*-
from pds.pre_processing import ViePreprocessor, EngPreprocessor
from pds.pre_processing.utils import remove_puntuation, split_para
from .doc2vec import SearchDoc2Vec


class CROffline():
    def __init__(self, collection, lang='en'):
        self.lang = lang
        self.doc2vec = SearchDoc2Vec(self.lang, collection)

    def retrieveCandidates(self, text, isPDF=False):
        # Output: [{input_para: string, candidate_list: [{title: string, content: [source_para: string]}]}]

        # Preprocess input text to word-paragraph
        input_paras = split_para(text, isPDF=isPDF)
        clean_paras = [remove_puntuation(para) for para in input_paras]

        if self.lang == 'en':
            words_paras_list = [EngPreprocessor.pp2word(
                par) for par in clean_paras]
        else:
            words_paras_list = [ViePreprocessor.pp2word(
                par) for par in clean_paras]

        # Search using Doc2vec
        search_result = self.doc2vec.search(words_paras_list)

        # Combine result
        result = [{'input_para': input_par, 'candidate_list': can_lst}
                  for input_par, can_lst in zip(input_paras, search_result)]

        return result


# Example Demo for Training Doc2Vec models for CROffline Module
# Please read the documentation before using!!!
# Note: All boolean parameter for each function in this demo is also its default.
if __name__ == '__main__':
    from pds.database import ExDatabase

    # Connect to DB
    database = ExDatabase(
        'mongodb+srv://phuockaus:phuockaus0412@pds.qfuxg.mongodb.net/myFirstDatabase?retryWrites=true&w=majority', 'Documents')

    # Get English Collection
    collection_en_cursor = database.getCollection('eng')
    collection_en = list(map(lambda item: item, collection_en_cursor))

    # Get Vietnamese Collection and preprocessing
    # collection_vi_cursor = database.getCollection('vie')
    # collection_vi = list(map(lambda item: item, collection_vi_cursor))

    text = """Cách truyền thống khi cải thiện hiệu suất của CNN là tăng sâu hoặc rộng của mô hình học sâu. Như mô hình ResNet mở rộng từ ResNet-18 lên ResNet-200 bằng cách sử dụng nhiều lớp hơn.

    Bài toán hiểu (Machine reading comprehension) không phải là  ra gần . Những năm 1977, Lehnert và các cộng sự xây dựng một hệ thống hỏi gọi là QUALM. Năm 1999, Hirschman và các cộng sự xây  mô hình với 60 văn cho tập phát triển và 60 đoạn văn cho kiểm tra. Độ chính xác của mô hình đạt từ 30% đến 40%. Những mô hình thời kì này thường xây dựng trên phương pháp thống kê. Tuy nhiên do việc thiếu tập dữ liệu để xây dựng, bài toán này đã bị bỏ quên trong một thời gian dài.

    Năm 2013, Richardson tạo ra tập dữ liệu MCTest [17] gồm hơn 500 đoạn văn và 2000 câu hỏi. Sau đó, nhiều nhà nghiên cứu dùng các mô hình học máy trên tập liệu này. Tuy nhiên các mô hình này xây dựng trên các nguyên tắc và tập dữ liệu MCTest chưa lớn. Để giải quyết vấn đề này, Hermann và các cộng sự [9] tạo ra một cách thức để tạo ra tập dữ liệu lớn để xây dựng mô hình đọc hiểu. Đồng thời họ còn phát triển các giải thuật dùng cơ chế attention trên mạng học sâu có thể đọc và trả lời câu hỏi với chỉ cần một chút hiểu biết về ngôn ngữ. Kể từ năm 2015, sự ra đời của tập dữ liệu lớn và mô hình mạng học sâu, bài toán đọc hiểu phát triển triển một cách nhanh chóng. Hình 2.1 chỉ ra số lượng các bài báo về lĩnh vực đọc hiểu từ năm 2013. Vì thế, 2 nhà nghiên cứu Quoc V.Le và Mingxing Tan cùng các cộng sự tại Google Research – Brain Team ra một giải pháp mang tên EfficientNet giúp cải thiện chính xác của mô hình và nhu cầu tính toán bằng cách mở rộng hiệu quả theo mọi hướng không chỉ sâu mà còn mở rộng (scaling) cả rộng và phân giải.
    """
    # croff = CROffline(collection_vi, lang='vi')
    # search_result = croff.retrieveCandidates(text, isPDF=False)

    # print("\n>>> Test for Vietnamese offline candidate retrieval...")
    # print(search_result)

    text = """
    In the modern age of the 4th technological revolution, electronic e-commerce has become an area that has an extremely important influence on the economic developments of the country. The development of e-commerce does not only make a lot of benefits for business but also provides some new values for companies and individuals.

    In order to approach and contribute to promoting the popularity of e-commerce. We have researched some background knowledge about e-commerce and we will develop an e-commerce website, which helps users to buy and sell a very popular product - “Shoes”.

    Nowadays, e-commerce is a wide topic. So, I would limit the subject is creating an e-commerce website that users focus on 1 or 2 types of product only.

    E-wallet payment is a payment method used so that buyer's money can reach the seller because of the popularity and reliability of e-wallet in the market. QR code technology is applied to control customer’s access to the store and for customer to make payment.
    """
    croff = CROffline(collection_en, lang='en')
    search_result = croff.retrieveCandidates(text, isPDF=False)

    print("\n>>> Test for English offline candidate retrieval...")
    print(search_result)
