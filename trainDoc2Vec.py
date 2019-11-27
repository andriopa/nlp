import logging
import pickle
from gensim import utils
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stop_words = set(stopwords.words('english'))
# print(stop_words)
#-----------

def preprocessFile(docs):


    for i, line in enumerate(docs):

        if i % 10000 == 0:
            logging.info("read {0} reviews".format(i))

        yield utils.simple_preprocess(line)


def trainDV(documents, tags):

    tags = [x.strip() for x in tags]

    for i, line in enumerate(tags):
        tags[i] = [x.strip() for x in line.split(',')]

    # filterdocs = []
    # for line in docs:
    #     filterdocs.append([w for w in line if not w in stop_words])

    docs = list(preprocessFile(documents))

    finalDoc = []

    for i, docs in enumerate(docs):
        finalDoc += [TaggedDocument(docs, tags[i])]

    # print(finalDoc)
    print(len(finalDoc))

    model = Doc2Vec(finalDoc, dm=1, alpha=0.025, vector_size=100, min_alpha=0.025, window=10, min_count=10)
    model.train(finalDoc, total_examples=len(finalDoc), epochs=100)

    # save the model to disk
    filename = 'model_dv12100-ep100.sav'
    pickle.dump(model, open(filename, 'wb'))

