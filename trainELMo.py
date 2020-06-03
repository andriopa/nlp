import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing
import tensorflow.keras as keras
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

url = "https://tfhub.dev/google/elmo/3"
embed = hub.Module(url)


#-----------------------------------------------#

fileDoc = 'balanceD100-1-ELMo.csv'
fileTag = 'balanceT100-1-ELMo.csv'


with open(fileDoc, 'r', encoding="utf-8") as fp:
    docs = fp.readlines()

with open(fileTag, 'r', encoding="utf-8") as fp:
    labels = fp.readlines()

#  --------------------------------------------- #
print(len(docs), len(labels))

d = docs[:10000] + docs[100000:101000]
del docs
print(len(d))
docs = d

l = labels[:10000] + labels[100000:101000]
del labels
print(len(l))
labels = l

print (len(docs), len(labels))

le = preprocessing.LabelEncoder()
le.fit(labels)

def encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)


x_enc = docs
y_enc = encode(le, labels)

x_train = np.asarray(x_enc[:10000])
y_train = np.asarray(y_enc[:10000])
print('enc_train:', len(x_train),len(y_train))

x_test = np.asarray(x_enc[10000:11000])
y_test = np.asarray(y_enc[10000:11000])
print('enc_test:', len(x_test), len(y_test))

from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

def ELMoEmbedding(docs):
    return embed(tf.squeeze(tf.cast(docs, tf.string)), signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(100, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with tf.Session() as session:
   K.set_session(session)
   session.run(tf.global_variables_initializer())
   session.run(tf.tables_initializer())
   history = model.fit(x_train, y_train, epochs=10, batch_size=32)
   model.save_weights('./elmo-model-9.h5')

model.summary()

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./elmo-model-12.h5')
    predicts = model.predict(x_test, batch_size=10)


y_test = decode(le, y_test)
y_preds = decode(le, predicts)
#for item in y_test:
#       print(item.strip())
#print(y_test)
print('-----')
for item in y_preds:
        print(item.strip())


from sklearn import metrics


print(metrics.confusion_matrix(y_test, y_preds))

print(metrics.classification_report(y_test, y_preds))

print(metrics.classification.accuracy_score(y_test, y_preds))




