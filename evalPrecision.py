import gzip
import gensim
import pickle
import logging
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

filename = 'model_dv12100-ep100.sav'
model = pickle.load(open(filename, 'rb'))
print(model)

# train_file = "train110.txt.gz"
# tag_file = "tag110.txt"
# train_file = "doc9.csv.gz"
# tag_file = "tag9.csv"


# train_file = "train1.txt.gz"
# tag_file = "tag1.txt"
# train_file = "test11.txt.gz"
# tag_file = "tag11.txt"
# train_file = "sm1.txt.gz"
# tag_file = "tagsm1.txt"

train_file = "balanceConvertD100-3.csv.gz"
tag_file = "balanceConvertT100-3.csv"


def read_input(input_file):

    logging.info("reading file {0}...this may take a while".format(input_file))

    with gzip.open(input_file, 'rb') as f:
        count = 0
        for i, line in enumerate(f):
            count = count + 1
            if i % 10000 == 0:
                logging.info("read {0} reviews".format(i))

            yield gensim.utils.simple_preprocess(line)


with open(tag_file) as f:
    tags = f.readlines()

tags = [x.strip() for x in tags]

for i, line in enumerate(tags):
    tags[i] = [x.strip() for x in line.split(';')]

x = []
for item in tags:
    x.append(len(item))
# print(x)
print('avgSuggestion ', sum(x)/len(x))
# print(len(x))

docs = list(read_input(train_file))
logging.info("Done reading data file")

# docs = docs[:10000]
# tags = tags[:10000]

metrics = []
for k in [1, 3, 10]:
    for l in range(36, 100, 4):
        l = l/100
        countAll, countSug, countNotSug, countCor = 0, 0, 0, 0
        list = []
        # l1, l2 = [], []
        for i, doc in enumerate(docs):
            vectorDoc = model.infer_vector(doc)
            # print('No:', i)
            # print('GT:', len(tags[i]), tags[i])
            num2 = k
            var = model.docvecs.most_similar([vectorDoc], topn=k)
            # print('Sug:', var)
            var1 = []
            for item in var:
                if item[1] > l:
                    var1.append(item)

            num1, check = 0, 0
            for tag in tags[i]:
                try:
                    if len(var1) != 0:
                        countAll += 1
                    for item in var1:
                        if item[0] == tag:
                            # print(item[0], 'ok')
                            countCor += 1
                            num1 += 1
                            break
                except:
                    pass

            if num2 != 0 :
                if len(var1) != 0:
                    countSug += 1
                # if check == 1:
                #     l1.append(num1)
                #     l2.append(num2)
                    num = num1/num2
                    list.append(num)
                    # print('Prec', num)
                    # print('\n')

                else:
                    countNotSug +=1

        if countCor != 0:
            print ('rank:', k, 'sim', l)
            print('All:', countAll)
            print('Match:', countCor)
            print('Not Sug:', countNotSug)
            pr = sum(list)/len(list)

            print('Precision', pr)
            # print(sum(l1), sum(l2))
            # print(sum(l1)/sum(l2))
            line = str(k) + ', ' + str(l) + ', ' + str(countSug) + ', ' + str(countNotSug) + ', ' + str(countAll) + ', ' + str(countCor) + ', ' + str(pr)
            metrics.append(line)
            print(line)
print(metrics)
with open('j61blAll.txt', 'w', encoding="utf-8") as f:
    for item in metrics:
        f.write("%s\n" % item)

