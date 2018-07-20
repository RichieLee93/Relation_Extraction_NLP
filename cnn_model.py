

from __future__ import print_function
from argparse import ArgumentParser, FileType
from stanfordcorenlp import StanfordCoreNLP

import re
import json
import logging
import numpy as np
import spacy
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Embedding, Conv1D, MaxPooling1D,concatenate
from keras.models import Model
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Merge
from keras.utils import to_categorical

import win_unicode_console
win_unicode_console.enable()

parser = ArgumentParser(description='')
parser.add_argument('instances', type=FileType('r'), metavar='<instances>', help='Relation extraction instances.')
A = parser.parse_args()
logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)
nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2017-06-09')
instances = [json.loads(line) for line in A.instances]
relation_type = [instance['relations'][0]['type'] for instance in instances if len(nlp.word_tokenize(instance['content'])) <= 40]
le = LabelEncoder()
le.fit(['Cause-Effect', 'Instrument-Agency', 'Product-Producer', 'Product-Producer', 'Content-Container',
        'Entity-Origin', 'Entity-Destination', 'Component-Whole', 'Member-Collection', 'Message-Topic', 'Other'])
label = le.transform(relation_type)

# text_selected = [instance['content'][instance['mentions'][0]['start_char']:instance['mentions'][1]['end_char']] for instance in instances]
text_selected = [instance['content'] for instance in instances if len(nlp.word_tokenize(instance['content'])) <= 40]
instance_selected = [instance for instance in instances if len(nlp.word_tokenize(instance['content']))<= 40]
print(len(text_selected))
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(text_selected)
sequences = tokenizer.texts_to_sequences(text_selected)
word_index =  tokenizer.word_index
data_we = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)



pos_tags =[]
for i in text_selected:
    pos_tag = []
    tags = nlp.pos_tag(i)
    for j in tags:
        pos_tag.append(j[1])
    pos_tags.append(pos_tag)
postag_set = ['CC', 'CD', 'DT', 'Ex', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
              'PDT', 'POS', 'PRP', 'PP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD',
              'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '#', '$', ',', '.', ':']
for i in range(len(pos_tags)):
    for j in range(len(pos_tags[i])):
        if pos_tags[i][j] in postag_set:
            pos_tags[i][j] = postag_set.index( pos_tags[i][j])+1
        else:
            pos_tags[i][j] = len(postag_Set) + 1

pos_tags = np.array(pos_tags)
pos_tags = pad_sequences(pos_tags,maxlen=MAX_SEQUENCE_LENGTH)
position_distances_m1 = []
position_distances_m2 = []
character_list = ['-','_']
for i in range(len(text_selected)):
    # print(text_selected[i])
    tokens = nlp.word_tokenize(text_selected[i])
    find_m1_text = []
    for t in re.finditer(instance_selected[i]['mentions'][0]['text'],text_selected[i]):
        if text_selected[i][t.span()[1]].isalpha() == False:
            if t.span()[0] == 0:
                find_m1_text.append(t.span())
            else:
                if text_selected[i][t.span()[0]-1].isalpha() == False and text_selected[i][t.span()[1]] not in character_list and text_selected[i][t.span()[0]-1] not in character_list:
                    find_m1_text.append(t.span())
    find_m1_index = []
    for m, a in enumerate(tokens):
        if a == instance_selected[i]['mentions'][0]['text'].split()[0]:
            find_m1_index.append(m)
    for f in find_m1_text:
        if f[0] == instance_selected[i]['mentions'][0]['start_char']:
            m1_index =find_m1_index[find_m1_text.index(f)]
    position_distance_m1 = []
    for j in range(len(tokens)):
        position_distance_m1.append(j-m1_index+40)
    position_distances_m1.append(position_distance_m1)


    find_m2_text = []
    for t in re.finditer(instance_selected[i]['mentions'][1]['text'],text_selected[i]):
        if text_selected[i][t.span()[1]].isalpha() == False:
            if t.span()[0] == 0:
                find_m2_text.append(t.span())
            else:
                if text_selected[i][t.span()[0]-1].isalpha() == False and text_selected[i][t.span()[1]] not in character_list and text_selected[i][t.span()[0]-1] not in character_list:
                    find_m2_text.append(t.span())

    find_m2_index = []
    for m, a in enumerate(tokens):

        if a == instance_selected[i]['mentions'][1]['text'].split()[0]:
            find_m2_index.append(m)
    for f in find_m2_text:
        if f[0] == instance_selected[i]['mentions'][1]['start_char']:
            m2_index =find_m2_index[find_m2_text.index(f)]

    position_distance_m2 = []
    for j in range(len(tokens)):
        position_distance_m2.append(j-m2_index+40)
    position_distances_m2.append(position_distance_m2)

position_distances_m1 = np.array(position_distances_m1)
position_distances_m2 = np.array(position_distances_m2)
position_distances_m1 = pad_sequences(position_distances_m1,maxlen=MAX_SEQUENCE_LENGTH)
position_distances_m2 = pad_sequences(position_distances_m2,maxlen=MAX_SEQUENCE_LENGTH)






split = 7858
x_train_we = data_we[:split]
x_val_we = data_we[split:]
x_train_POS = pos_tags[:split]
x_val_POS = pos_tags[split:]
x_train_PositionDist1 = position_distances_m1[:split]
x_val_PositionDost1 = position_distances_m1[split:]
x_train_PositionDist2 = position_distances_m2[:split]
x_val_PositionDost2 = position_distances_m2[split:]
y_train = label[:split]
y_train = to_categorical(y_train)
y_val = label[split:]
# y_val = to_categorical(y_val)


embeddings_index = {}
with open('C:/Users/NLP/Documents/GitHub/relation-extraction/i2r/relation_extraction/glove.6B.300d.txt',encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer_we = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
embedding_layer_POS = Embedding(42 + 1,
                            3,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
embedding_layer_PositionDist1 = Embedding(80 + 1,
                            3,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
embedding_layer_PositionDist2 = Embedding(80 + 1,
                            3,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
print('sequence_input.shape',sequence_input.shape)
embedded_sequences_we = embedding_layer_we(sequence_input)
POS_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
print('POS_input.shape',POS_input.shape)
embedded_sequences_POS = embedding_layer_POS(POS_input)
PositionDist1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
print('PositionDist1_input.shape',PositionDist1_input.shape)
embedded_sequences_PositionDist1 = embedding_layer_PositionDist1(PositionDist1_input)
PositionDist2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
print('PositionDist2_input.shape',PositionDist2_input.shape)
embedded_sequences_PositionDist2 = embedding_layer_PositionDist2(PositionDist2_input)
x = concatenate([embedded_sequences_we,embedded_sequences_POS,embedded_sequences_PositionDist1, embedded_sequences_PositionDist2])
print('0',x.shape)
x = Conv1D(128, 5, activation='relu')(x)
print('1',x.shape)
x = MaxPooling1D(5)(x)
print('2',x.shape)
x = Conv1D(128, 5, activation='relu')(x)
print('3',x.shape)
#x = MaxPooling1D(5)(x)
# print('4',x.shape)
# x = Conv1D(128, 5, activation='relu')(x)
# print('5',x.shape)
x = MaxPooling1D(5)(x)  # global max pooling
print('6',x.shape)
x = Dropout(1)(x)
x = Flatten()(x)
print('7',x.shape)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
print('8',x.shape)
preds = Dense(10, activation='softmax')(x)

model = Model(inputs=[sequence_input,POS_input,PositionDist1_input,PositionDist2_input], outputs=preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit([x_train_we,x_train_POS,x_train_PositionDist1,x_train_PositionDist2], y_train, validation_split=0.15,
          epochs=50, batch_size=8)
# score = model.evaluate(x_train_we, y_train, verbose=0)
# print('train score:', score[0])
# print('train accuracy:', score[1])
# score = model.evaluate([x_val_we,x_val_POS,x_val_PositionDost1,x_val_PositionDost2], y_val, verbose=0)
# print('Validation score:', score[0])
# print('Validation accuracy:', score[1])










# data = np.load('C:/Users/NLP/Documents/GitHub/relation-extraction/data/data.npy')
# # split the data into a training set and a validation set
# labels = np.load('C:/Users/NLP/Documents/GitHub/relation-extraction/data/label.npy')
#
# for i in range(len(data)):
#     if len(list(data[i])) < 97:
#         temp = list(data[i])
#         for m in range(len(data[i]) + 1, 98):
#             temp.append([0] * 343)
#         data[i] = np.array(temp)
#     for j in range(len(data[i])):
#         data[i][j] = list(data[i][j])
# # for i in range(len(data)):
# #     data[i] = sequence.pad_sequences(data[i],padding='post')
# # np.save('C:/Users/NLP/Documents/GitHub/relation-extraction/data/data_new.npy',data)
# data = np.array(list(data))
# print(data.shape)
# #
# x_train = data[:6000] # training set
#
# y_train = labels[:6000]  # training label
# y_train = to_categorical(y_train)
# print(y_train.shape)
# x_val = data[6000:]  # validation set
# y_val = labels[6000:]  # validation label
# y_val = to_categorical(y_val)
# print(y_val.shape)
# # train a 1D convnet with global maxpoolinnb_wordsg
#
# model = Sequential()
#
# model.add(Conv1D(128, 5, activation='tanh',input_shape=(data.shape[1],data.shape[2])))
# model.add(MaxPooling1D(5,padding='same'))
# # model.add(Conv1D(128, 5, activation='tanh'))
# # model.add(MaxPooling1D(5,padding='same'))
# # model.add(Conv1D(128, 5, activation='tanh'))
# # model.add(MaxPooling1D(35,padding='same'))
# model.add(Flatten())
# model.add(Dense(128, activation='tanh'))  # fully connected layer
# model.add(Dense(10, activation='softmax'))  # softmax
#
# # optimization
# model.compile(loss='categorical_crossentropy',
#               optimizer='Adadelta',
#               metrics=['accuracy'])
#
# #RMSprop
# model.fit(x_train, y_train, nb_epoch=50,batch_size=8, validation_split=0.2)
#
# score = model.evaluate(x_train, y_train, verbose=0)
# print('train score:', score[0])
# print('train accuracy:', score[1])
# score = model.evaluate(x_val, y_val, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
