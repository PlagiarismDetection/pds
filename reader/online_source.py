import os
import requests
from bs4 import BeautifulSoup
from tika import parser
import re
from pds.pre_processing.utils import split_para

class OnlSource():
    def __init__(self, metadata, content):
        self.url = metadata['url']
        self.title = metadata['title']
        self.snippet = metadata['snippet']
        self.content = content

    def getUrl(self):
        return self.url

    def getTitle(self):
        return self.title

    def getSnippet(self):
        return self.snippet

    def getContent(self):
        return self.content

class ReadOnlSource():
    def __init__(self):
        pass

    @staticmethod
    def download_pdf_from_url(url, output_dir=''):
        # Max 10 seconds to connect to server and max 20 seconds to wait on response
        response = requests.get(url, timeout=10)
        print(response.headers)

        if response.status_code == 200:
            file_path = os.path.join(output_dir, os.path.basename(url))
            print('>>> Downloading ', file_path, '...')
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print('>>> Finish')
            return file_path

        # print('Cant download from ', url)
        return False

    @classmethod
    def read_pdf_from_url(cls, url):
        path = cls.download_pdf_from_url(url)
        if path:
            raw = parser.from_file(path)
            os.remove(path)
            return raw["content"]
        return False
        # print('Cant read ', url)

    @staticmethod
    def read_text_from_url(url):
        # Max 5 seconds to connect to server and max 10 seconds to wait on response
        response = requests.get(url, verify=False, timeout=10)
  
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, features="html.parser")
        
            # kill all script and style, math elements
            for script in soup(["script", "style","math"]):
                script.extract()    # rip it out

            # Get all text in p tag
            texts = []
            all_para = soup.body.find_all("p")
            for para in all_para:
                handle_text = re.sub(' +', ' ',para.text.strip())
                remove_cite = re.sub('\[.*?\]', '',handle_text) #example [1][2]...
                texts.append(remove_cite)

            # drop blank lines
            content = '\n'.join(text for text in texts if text)
            return content
        return False

    @staticmethod
    def is_pdf_url(url):
        if url.find('?') > 0:
            url = url[:url.find('?')]
        return url.endswith('.pdf')

    @staticmethod
    def handle_special_url(url):
        # Arxiv
        if re.search("arxiv.org/abs",url):
            url=re.sub("arxiv.org/abs", "arxiv.org/pdf",url)+'.pdf'
        # Researchgate
        elif re.search("researchgate.net/publication", url):
            response = requests.get(url, verify=False)
            soup = BeautifulSoup(response.content, "html.parser")
            child_soup = soup.find_all('a')
            text = "Download full-text PDF"
            for i in child_soup:
                if i.text == text:
                    url = i['href']
        return url

    @classmethod
    def read_onl(cls, url):
        content = ''
         # Handle link of some scientific article
        url = cls.handle_special_url(url)
        try:
            if cls.is_pdf_url(url):
                print('>>> Read from PDF:', url)
                content = split_para(cls.read_pdf_from_url(url), isPDF=True)
            else:
                print('>>> Read from URL:', url)
                content = split_para(cls.read_text_from_url(url), isPDF=False)
        except Exception as e:
            print('>>> Cant read url: ', url)
            print(e)
        return content

    
    @classmethod
    def getOnlList_PP(cls, searchlist_pp):
        onlList = []
        candidate_list = searchlist_pp['candidate_list']
        for can in candidate_list:
            content = cls.read_onl(can['url'])
            if content:
                can['content'] = content
                onlList.append(can)
        searchlist_pp['candidate_list'] = onlList
        print('Read ',len(searchlist_pp['candidate_list']), 'valid sources')
        return searchlist_pp

    @classmethod
    def getOnlList_Doc(cls, search_result):
        for searchres_pp in search_result:
            cls.getOnlList_PP(searchres_pp)
        print('Read Online source Finish')
        return search_result
