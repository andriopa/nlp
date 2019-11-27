import os
from connectFTP import *
from readXML import *
from trainDoc2Vec import *


# Main Function
if __name__ == "__main__":

    documents, tags = [], []

    for i in range(1):

        digit = str(i).zfill(2)

        filename = 'pubmed/pubmed19n00' + digit + '.xml.gz'

        getFile(filename)

        print(filename)

        document, tag = parse(filename)

        documents.extend(document)

        tags.extend(tag)

        print(len(documents))  # ,'--',len(tags)

        os.remove(filename)


    # write results in csv file
    with open('export.csv', 'w', encoding="utf-8") as f:
        for item in documents:
            f.write("%s\n" % item)

    # # write results in csv file
    with open('tag.csv', 'w', encoding="utf-8") as f:
        for item in tags:
            f.write("%s\n" % item)


    # trainDV(documents,tags)