from tika import parser
from abc import ABC
import glob


def getPDFList(folder):
    fileList = glob.glob('{}/*.pdf'.format(folder))
    dataList = list(map(lambda file: parser.from_file(file), fileList))
    pdfList = []
    for data in dataList:
        pdf = PDF(data['metadata'], data['content'])
        pdfList.append(pdf)
    return pdfList


def pdfClean(lst):
    try:
        for pdf in lst:
            del pdf
    except:
        print('Type Error')


class PDF(ABC):
    def __init__(self, metadata, content):
        super().__init__()
        self.filename = metadata['resourceName'].encode('latin1').decode(
            'unicode_escape').encode('latin1').decode('utf8')[2:-1] if 'resourceName' in metadata.keys(
        ) else ''
        self.title = metadata['resourceName'].encode('latin1').decode(
            'unicode_escape').encode('latin1').decode('utf8')[2:-1] if 'resourceName' in metadata.keys(
        ) else ''
        self.author = metadata['Author'] if 'Author' in metadata.keys() else ''
        self.creation_date = metadata['Creation-Date'] if 'Creation-Date' in metadata.keys() else ''
        self.content = content
