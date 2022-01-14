from abc import ABC
from pds.reader import getPDFList, pdfClean, getDOCXList, docxClean
from pymongo import MongoClient


class Database(ABC):
    # module database manages basic manipulations on the local database
    def __init__(self, CONNECTION_STRING, dbname):
        client = MongoClient(CONNECTION_STRING)
        self.defaultFields = ['_id', 'Filename', 'Title',
                              'Author', 'Creation-date', 'Content']
        self.db = client[dbname]

    def insertByFile(self, colname, folder):
        # -> None: insert a document in file-type
        self.__pdf_push(folder, colname)
        self.__docx_push(folder, colname)

    def insertByData(self, colname, filename, title, author, creation_date, content):
        # -> Dictionary: insert a document in text-type
        collection_name = self.db[colname]
        data = self.createDocuments(
            filename, title, author, creation_date, content)
        return collection_name.insert_one(data)

    def getCollection(self, colname):
        # -> list(dictionary): return a list of documents in a collection
        collection = self.db[colname]
        return collection.find()

    def updateDocuments(self, colname, filter={}, updateValue=None):
        # -> list(dictionary): return a list of updated documents in a collection
        # Note: only update on the basic existed fields.
        if len([field for field in updateValue.keys() if field in self.defaultFields]) == len(updateValue.keys()):
            return self.db[colname].update_many(filter, {'$set': updateValue})
        return False

    def deleteDocuments(self, colname, filter={}):
        # -> list(dictionary): return a list of deleted documents in a collection
        return self.db[colname].delete_many(filter)

    @staticmethod
    def createDocuments(filename, title, author, creation_date, content):
        item = {'Filename': filename,
                'Title': title,
                'Author': author,
                'Creation-date': creation_date,
                'Content': content}
        return item

    def __pdf_push(self, folder, colname):
        pdf_data = getPDFList(folder)
        pdf_docs = list(
            map(lambda data: self.createDocuments(data.filename, data.title, data.author, data.creation_date, data.content), pdf_data))
        if(pdf_docs != []):
            collection_name = self.db[colname]
            collection_name.insert_many(pdf_docs)
        pdfClean(pdf_data)

    def __docx_push(self, folder, colname):
        docx_data = getDOCXList(folder)
        docx_docs = list(
            map(lambda data: self.createDocuments(data.filename, data.title, data.author, data.creation_date, data.content), docx_data))
        if(docx_docs != []):
            collection_name = self.db[colname]
            collection_name.insert_many(docx_docs)
        docxClean(docx_data)
