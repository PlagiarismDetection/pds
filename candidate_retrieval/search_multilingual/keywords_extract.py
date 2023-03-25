from keybert import KeyBERT


class KeyWordsExtract():
    def __init__(self, lang='EN'):
        if lang == 'EN':
            # author suggest using this model for English version
            self.model = 'all-MiniLM-L6-v2'
        else:
            # author suggest using this model for multiligual version
            self.model = 'paraphrase-multilingual-MiniLM-L12-v2'

        self.kw_model = KeyBERT(model=self.model)

    def extractKeyWords(self, document):
        # keyphrase_ngram_range: the number of result keywords e.g. 3 means each keyword should has length of 3
        # use_mmr (Maximal Marginal Relevance) which based on cosine similarity, high diversity
        probabilityKeyWords = self.kw_model.extract_keywords(document, keyphrase_ngram_range=(3, 3), use_mmr=True, diversity=0.7)

        # ATM, I will ignore the percentage score of each keyword
        return map(lambda probabilityKeyWord: probabilityKeyWord[0], probabilityKeyWords)

    def extractClusterKeyWords(self, cluster):
        # keywords = cluster.map(lambda paragraph: self.extractKeyWords(paragraph))

        keywords = [] # Set data structure
        for paragraph in cluster:
            keywords.extend(self.extractKeyWords(paragraph))

        # prevent duplicated elemennts
        return list(set(keywords))



# test
   
# x1 = KeyWordsExtract()

# cluster1 = ['''Mangrove forests, found at the edge of tropical and subtropical coastlines, are nutrient-rich breeding grounds for myriad species. Fish, birds, mammals and reptiles can all be found here – and the maze of twisted, stilted tree roots protects against predators, making them ideal nurseries. From Everglades National Park in Florida, which is home to threatened species of birds and amphibians, to the Australian mangroves where over 100 species of molluscs are found, to the Caribbean mangroves where rare green sea turtles dwell, these sea forests provide a critical habitat for many species.''',
#            '''The marine life – crustaceans, prawns, lobsters, crab and fish – that thrives in mangrove ecosystems supports local fisheries, providing food and revenue for coastal communities. The areas surrounding some of the forests are popular destinations for ecotourists, and revenue from activities such as birdwatching, kayaking and fishing can give an additional boost to local economies. But aside from providing a home for marine life and supporting people’s livelihoods, mangrove forests protect the structure of the coastline itself. The roots of the trees filter the water by trapping sediment, which slows coastal erosion, stabilises the shore, and stops sediment from damaging coral reefs and seagrass meadows.''',
#            '''Believe it or not, resurrected mangrove forests have all-but ended criminal activity in parts of Kenya. By 2009, some areas had lost over 80% of their mangroves, causing a reduction in the fish stocks that villagers relied on for their livelihoods. Without fish, locals in places such as Gazi turned to logging as well as illegal poaching of elephant and rhino. Today, however, a programme of mangrove reforestation has created new and legal financial opportunities, resulting in criminal activity in the region dropping by 90\% over a six-year period, according to the Kenya Wildlife Service.'''
#         ]

# cluster2 = []

# print(x1.extractClusterKeyWords(cluster1))
# print(x1.extractClusterKeyWords(cluster2))
