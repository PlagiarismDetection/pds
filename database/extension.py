from .database import Database


class ExDatabase(Database):
    def __init__(self, CONNECTION_STRING, dbname, colname='vie'):
        super().__init__(CONNECTION_STRING, dbname)
        fields = self.db[colname].find_one(
            {}).keys() if colname in self.db.list_collection_names() else []
        self.extensionFields = [
            field for field in fields if field not in self.defaultFields]

    def addField(self, colname, fieldname, defaultValue=None):
        # -> list(dictionary): return a list of documents with new field in a collection
        self.extensionFields.append(fieldname)
        return self.updateDocuments(colname, updateValue={fieldname: defaultValue})

    def updateFieldBasedOnContent(self, colname, fieldname, func):
        # -> None: update a field value base on another field(s)
        docs = self.db[colname].find({})
        for doc in docs:
            id = doc['_id']
            content = doc['Content']
            self.updateDocuments(colname, filter={'_id': id}, updateValue={
                                 fieldname: func(content)})
        self.extensionFields.append(fieldname)

    def removeField(self, colname, fieldname):
        # -> list(dictionary): return a list of documents that had been deleted in a collection
        # Note: only delete an extension fields.
        if fieldname in self.extensionFields:
            self.extensionFields.remove(fieldname)
            return self.db[colname].update_many({}, {'$unset': {fieldname: 1}})
        return False

    def updateDocuments(self, colname, filter={}, updateValue=None):
        # -> list(dictionary): return a list of documents that had been updated in a collection
        # Note: only update on existed fields.
        if len([field for field in updateValue.keys() if field in self.defaultFields or field in self.extensionFields]) == len(updateValue.keys()):
            return self.db[colname].update_many(filter, {'$set': updateValue})
        return False

    def createDocuments(self, filename, title, author, creation_date, content):
        item = super().createDocuments(filename, title, author, creation_date, content)
        for field in self.extensionFields:
            item[field] = None
        return item
