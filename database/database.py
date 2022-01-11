from abc import ABC
from pds.reader import getPDFList, pdfClean, getDOCXList, docxClean
from pds.pre_processing.eng_preprocessing import EngPreprocessing
from pds.pre_processing.vnm_preprocessing import VnmPreprocessing
from pymongo import MongoClient


class Database(ABC):
    def __init__(self, CONNECTION_STRING, dbname):
        client = MongoClient(CONNECTION_STRING)
        self.db = client[dbname]

    def pushByFile(self, folder, collection):
        self.__pdf_push(folder, collection)
        self.__docx_push(folder, collection)

    def pushByData(self, filename, title, author, creation_date, content, collection):
        collection_name = self.db[collection]
        data = self.__createDocuments(
            filename, title, author, creation_date, content, collection)
        collection_name.insert_one(data)

    @staticmethod
    def __createDocuments(filename, title, author, creation_date, content, collection):
        content_w = EngPreprocessing.preprocess2word(
            content) if collection == 'eng' else VnmPreprocessing.preprocess2word(content)
        content_s = EngPreprocessing.preprocess2sent(
            content) if collection == 'eng' else VnmPreprocessing.preprocess2sent(content)
        item = {'Filename': filename,
                'Title': title,
                'Author': author,
                'Creation-date': creation_date,
                'Content': content,
                'Content-word': content_w,
                'Content-sent': content_s}
        return item

    @classmethod
    def __pdf_push(cls, folder, collection):
        pdf_data = getPDFList(folder)
        pdf_docs = list(
            map(lambda data: cls.__createDocuments(data.filename, data.title, data.author, data.creation_date, data.content, collection), pdf_data))
        if(pdf_docs != []):
            collection_name = cls.db[collection]
            collection_name.insert_many(pdf_docs)
        pdfClean(pdf_data)

    @classmethod
    def __docx_push(cls, dbname, folder, collection):
        docx_data = getDOCXList(folder)
        docx_docs = list(
            map(lambda data: cls.__createDocuments(data.filename, data.title, data.author, data.creation_date, data.content, collection), docx_data))
        if(docx_docs != []):
            collection_name = dbname[collection]
            collection_name.insert_many(docx_docs)
        docxClean(docx_data)

    @staticmethod
    def getCollection(dbname, colname):
        collection = dbname[colname]
        return collection.find()
