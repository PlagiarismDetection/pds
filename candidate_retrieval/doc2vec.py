import time
from abc import ABC
import numpy as np
import faiss
import pickle

# gensim modules
import gensim
from gensim import utils
from gensim.models import Doc2Vec

from pds.pre_processing import ViePreprocessor
from pds.pre_processing import EngPreprocessor

from pds.database import ExDatabase
from django.conf import settings

RELATIVE_PATH = "./pds/candidate_retrieval"


class SearchDoc2Vec(ABC):
    def __init__(self, lang, collection):
        self.lang = lang
        self.collection = collection
        self.preprocessor = ViePreprocessor if lang == 'vi' else EngPreprocessor

        self.model = self.get_model()
        self.faiss = self.get_faiss()

    def __training_model(self, train_corpus, vector_size=500, min_count=1, epochs=150):
        # Initilize model
        model = gensim.models.doc2vec.Doc2Vec(
            vector_size=vector_size, min_count=min_count, epochs=epochs)

        # Build a vocabulary
        print("Build Vocab...")
        model.build_vocab(train_corpus)

        # Training model
        print("Training model...")
        model.train(train_corpus, total_examples=model.corpus_count,
                    epochs=model.epochs)

        # Saving model
        model.save(f'{RELATIVE_PATH}/model/{self.lang}/{self.lang}.d2v')

        return model

    def evaluate_model(self):
        start_time = time.time()
        print("Evaluating model...")

        #  Get Vector-para in MongoDB
        para_vector_list = [
            par_vec for d in self.collection for par_vec in d['Vector-para']]

        # Infer all paragraphs to vectors
        word_para_list = [
            word_para for doc in self.collection for word_para in doc['Content-word-para']]
        print("Length list: ", len(word_para_list))
        test_vecs = [self.model.infer_vector(
            word_para) for word_para in word_para_list]
        print("Infer vector done")

        # Step 1: Change data type
        embeddings = np.asarray(
            [embedding for embedding in para_vector_list]).astype("float32")

        # Step 2: Instantiate the index
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Step 3: Pass the index to IndexIDMap
        index = faiss.IndexIDMap(index)

        # Step 4: Add vectors and their IDs
        ids = np.asarray(range(len(embeddings))).astype("int64")
        index.add_with_ids(embeddings, ids)

        faiss.write_index(index, 'test.index')

        accuracy = 0.0
        for source_id, test_vec in enumerate(test_vecs):

            # Retrieve the 5 nearest neighbours
            D, I = index.search(np.array([test_vec]), k=5)

            if I[0][0] == source_id:
                accuracy += 1
            # else:
            #     print("Sentence ", source_id)
            #     print(I)
            #     print(D)
            #     print()

        accuracy /= len(para_vector_list)

        print("Accuracy: ", accuracy)
        print("--- %s seconds ---" % (time.time() - start_time))

        return accuracy

    def __read_wiki_corpus(self):
        para_list = []
        try:
            with open(f"{RELATIVE_PATH}/model/{self.lang}/para_list_{self.lang}.txt", "rb") as file:   # Unpickling
                para_list = pickle.load(file)
        except:
            with open(f'{RELATIVE_PATH}/model/dataset/{self.lang}_wiki_para.txt', mode='r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    para_list.append(line)

            # Save para_list
            with open(f"{RELATIVE_PATH}/model/{self.lang}/para_list_{self.lang}.txt", "wb") as file:  # Pickling
                pickle.dump(para_list, file)

        return para_list

    def __preprocessing(self, para_list):
        pp_train = []

        try:
            with open(f"{RELATIVE_PATH}/model/{self.lang}/pp_train_{self.lang}.txt", "rb") as file:   # Unpickling
                pp_train = pickle.load(file)
        except:
            print("   Get Content-para From Collection...")
            pp_train = [
                word_para for d in self.collection for word_para in d['Content-word-para']]

            print("   Preprocess Wiki Para corpus...")
            pp_train += [self.preprocessor.pp2word(
                para, replace_num=None, lowercase=True) for para in para_list]

            # Save pp_train
            with open(f"{RELATIVE_PATH}/model/{self.lang}/pp_train_{self.lang}.txt", "wb") as file:  # Pickling
                pickle.dump(pp_train, file)

        return pp_train

    def create_train_corpus(self):
        print("Read Wiki Para corpus...")
        para_list = self.__read_wiki_corpus()

        # Get preprocessed words list for each para for train corpus
        print("Preprocessed para list...")
        pp_train = self.__preprocessing(para_list)

        # Create Train Corpus by TaggedDocument obj
        print("Create Train Corpus...")
        train_corpus = list([gensim.models.doc2vec.TaggedDocument(
            doc, [i]) for i, doc in enumerate(pp_train)])

        return train_corpus

    def update_vector_para_field(self):
        def addVectorParaField(word_para_list):
            vector_para_list = [self.model.infer_vector(
                par).tolist() for par in word_para_list]
            return vector_para_list

        #  Connect MongoDB
        database = ExDatabase(settings.MONGODB_HOST, 'Documents')

        # Vietnamese
        database.addField(self.lang, 'Vector-para')
        database.updateFieldBasedOnField(
            self.lang, 'Vector-para', addVectorParaField, otherField='Content-word-para')

        return

    def get_model(self):
        # Try load from saved model, else training model
        try:
            print("Load Model...")
            model = Doc2Vec.load(
                f'{RELATIVE_PATH}/model/{self.lang}/{self.lang}.d2v')
        except:
            print("Training Model...")

            #  Read Training dataset to create train_corpus
            train_corpus = self.create_train_corpus()

            # Training
            start_time = time.time()
            model = self.__training_model(
                train_corpus, vector_size=100, min_count=2, epochs=50)
            print("Trained done!")
            print("--- %s seconds ---" % (time.time() - start_time))

            #  Infer all offline paragraphs and update on MongoDB
            self.update_vector_para_field()

        return model

    def get_faiss(self):
        print("Get Faiss Search...")
        #  Get para vector and mapping list in collection
        para_vector_list = self.get_para_vector_list()

        # Step 1: Change data type
        embeddings = np.asarray(
            [embedding for embedding in para_vector_list]).astype("float32")

        # Step 2: Instantiate the index
        # IndexFlatIP: Inner Product => cosine similarity_score
        # IndexFlatL2: Euclidian distance
        index = faiss.IndexFlatIP(embeddings.shape[1])

        # Step 3: Pass the index to IndexIDMap
        index = faiss.IndexIDMap(index)

        # Step 4: Normalize embedding vectors for calculate cosine sim
        faiss.normalize_L2(embeddings)

        # Step 5: Add vectors and their IDs
        ids = np.asarray(range(len(embeddings))).astype("int64")
        index.add_with_ids(embeddings, ids)

        return index

    def get_para_vector_list(self):
        para_vector_list = [
            par_vec for d in self.collection for par_vec in d['Vector-para']]

        return para_vector_list

    def get_mapping_para_vector_list(self):
        mapping_to_docid = [i for docid, doc in enumerate(
            self.collection) for i in [docid]*len(doc['Vector-para'])]
        mapping_to_parid = [
            i for doc in self.collection for i in range(len(doc['Vector-para']))]

        return mapping_to_docid, mapping_to_parid

    def search(self, words_paras_list):
        #  Get mapping list
        mapping_to_docid, mapping_to_parid = self.get_mapping_para_vector_list()

        # Infer all paragraphs to vectors
        search_vectors = [self.model.infer_vector(
            par) for par in words_paras_list]

        # Normalize L2
        search_vectors = [np.array(vec)/np.linalg.norm(vec)
                          for vec in search_vectors]
        
        result = []

        for vec in search_vectors:
            top_cos_sim, top_id = self.faiss.search(np.array([vec]), k=5)

            sim_pars = []
            # print(top_sims[0])

            for id, cos_sim in zip(top_id[0], top_cos_sim[0]):
                if cos_sim >= 0.6:
                    tag_doc_id = mapping_to_docid[id]
                    tag_para_id = mapping_to_parid[id]

                    doc = self.collection[tag_doc_id]

                    for candidate in sim_pars:
                        if candidate['title'] == doc["Title"]:
                            candidate['content'].append(
                                doc["Content-para"][tag_para_id])
                            break
                    else:
                        sim_pars.append({
                            'title': doc["Title"],
                            'content': [doc["Content-para"][tag_para_id]],
                            'sm': cos_sim
                        })

            # print(sim_pars[0]['content'])
            result += [sim_pars]

        return result
