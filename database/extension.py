from .database import Database


class ExDatabase(Database):
    def __init__(self, CONNECTION_STRING, dbname, colname='vie'):
        super().__init__(CONNECTION_STRING, dbname)
        fields = self.db[colname].find_one({}).keys()
        self.extensionFields = [
            field for field in fields if field not in self.defaultFields]

    def updateFieldBasedOnContent(self, colname, fieldname, func):
        docs = self.db[colname].find({})
        for doc in docs:
            id = doc['_id']
            content = doc['Content']
            self.updateDocuments(colname, filter={'_id': id}, updateValue={
                                 fieldname: func(content)})
        self.extensionFields.append(fieldname)

    def removeField(self, colname, fieldname):
        if fieldname in self.extensionFields:
            self.extensionFields.remove(fieldname)
            return self.db[colname].update_many({}, {'$unset': {fieldname: 1}})
        return False

    def createDocuments(self, filename, title, author, creation_date, content):
        item = super().createDocuments(filename, title, author, creation_date, content)
        for field in self.extensionFields:
            item[field] = None
        return item
