import docx
import glob
from abc import ABC


def getDOCXList(cls, folder):
    fileList = glob.glob('{}/*.docx'.format(folder))
    dataList = list(map(lambda filename: docx.Document(filename), fileList))
    docxList = []
    for data in dataList:
        doc = DOCX(data.core_properties, getDocxText(data))
        docxList.append(doc)
    return docxList


def getDocxText(obj):
    docx_paras = obj.paragraphs
    full = []
    for para in docx_paras:
        full.append(para.text)
    return '\n'.join(full)


def docxClean(lst):
    try:
        for doc in lst:
            del doc
    except:
        print('Type Error')


class DOCX(ABC):
    def __init__(self, metadata, content):
        super().__init__()
        self.filename = metadata.title
        self.title = metadata.title
        self.author = metadata.author
        self.content = content
        self.creation_date = metadata.created
