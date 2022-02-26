import time
from abc import ABC

# gensim modules
import gensim
from gensim import utils
from gensim.models import Doc2Vec


class SearchDoc2Vec(ABC):
    @classmethod
    def __training_model(cls, train_corpus, lang, vector_size=500, min_count=1, epochs=150):
        start_time = time.time()

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

        print("Trained done!")
        print("--- %s seconds ---" % (time.time() - start_time))

        # Evaluating model
        # start_time = time.time()
        # print("Evaluating model...")
        # print("Accuracy: ", cls.__evaluate_model(model, train_corpus))
        # print("--- %s seconds ---" % (time.time() - start_time))

        # Saving model
        model.save('./checker/model/' + lang + '.d2v')

        return model

    @classmethod
    def __evaluate_model(cls, model, train_corpus):
        ranks = []
        for doc_id in range(len(train_corpus)):
            inferred_vector = model.infer_vector(train_corpus[doc_id].words)
            sims = model.docvecs.most_similar([inferred_vector], topn=1)
            ranks.append(sims)

        accuracy = 0.0
        for par_id in range(len(train_corpus)):
            if ranks[par_id][0][0] == par_id:
                accuracy += 1

        accuracy /= len(train_corpus)
        return accuracy


    @classmethod
    def get_model(cls, collection, lang):
        # Try load from saved model, else training model
        try:
            model = Doc2Vec.load('./checker/model/' + lang + '.d2v')
        except:
            # Get preprocessed words list for each para for train corpus
            pp_train = [word_para for d in collection for word_para in d['Content-word-para']]

            # Create tag for each train para corpus is title of its para
            tag_title = []
            tag_para_id = []

            for d in collection:
                tag_title += [d['Title']]*len(d['Content-para'])
                tag_para_id += list(range(len(d['Content-para'])))

            # Create Train Corpus by TaggedDocument obj
            train_corpus = list([gensim.models.doc2vec.TaggedDocument(
                doc, [(tag_title[i], tag_para_id[i])]) for i, doc in enumerate(pp_train)])

            model = cls.__training_model(train_corpus, lang)

        return model

    @classmethod
    def search(cls, model, words_paras_list, collection):
        # Infer all paragraphs to vectors
        inferred_vectors = [model.infer_vector(
            par) for par in words_paras_list]

        result = []

        for vec in inferred_vectors:
            top_sims = model.docvecs.most_similar([vec], topn=5)
            sim_pars = []
            print(top_sims[0]) 

            for tag, sm in top_sims:
                if sm >= 0.3:
                    tag_title, tag_para_id = tag
                    doc = next(
                        (d for d in collection if d['Title'] == tag_title), None)

                    for candidate in sim_pars:
                        if candidate['title'] == doc["Title"]:
                            candidate['content'].append(
                                doc["Content-para"][tag_para_id])
                            break
                    else:
                        sim_pars.append({
                            'title': doc["Title"],
                            'content': [doc["Content-para"][tag_para_id]],
                            'sm': sm
                        })

            print(sim_pars[0]['content'])
            result += [sim_pars]
        
        return result
