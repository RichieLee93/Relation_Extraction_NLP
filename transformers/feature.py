__all__ = ['FeaturesTransformer']

import numpy as np

from ycml.transformers import PureTransformer
from ..utils import find_token_span
import spacy


class FeaturesTransformer(PureTransformer):
    def __init__(self, **kwargs):
        kwargs.setdefault('nparray_dtype', np.object)
        super().__init__(**kwargs)
    #end def

    def transform_one(self, relation_instance):
        nlp = spacy.load('en_vectors_web_lg')
        m1_dict, m2_dict = relation_instance['paired_mentions']['m1'], relation_instance['paired_mentions']['m2']

        corenlp_annotations = relation_instance['corenlp_annotations']
        sentences = corenlp_annotations['sentences']

        m1_span = find_token_span(corenlp_annotations, m1_dict)
        m2_span = find_token_span(corenlp_annotations, m2_dict)

        #generate word embedding feature vector
        vector_sum = [nlp.vocab[tok['lemma']].vector for tok in sentences[m1_span[1][0]]['tokens'][0:len(sentences[m1_span[1][0]]['tokens'])]]

        #generate pos tag feature vector
        pos_vecs = []
        pos_sum = [tokens['pos'] for tokens in sentences[m1_span[1][0]]['tokens']]
        postag_set = ['CC','CD','DT','Ex','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS',
                      'PDT','POS','PRP','PP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD',
                      'VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','#','$',',','.',':']
        for i in range(len(pos_sum)):
            pos_vec = [0]*len(postag_set)
            for j in range(len(postag_set)):
                if postag_set[j] == pos_sum[i]:
                    pos_vec[j] = 1
            pos_vecs.append(np.array(pos_vec))

        #generate mention marks feature vector
        mark = []
        for i in range(len(sentences[m1_span[1][0]]['tokens'])):
            men_mark = [0, 0]
            if i in range(m1_span[0][1], m1_span[1][1] + 1):
                men_mark = [1, 0]
            if i in range(m2_span[0][1], m2_span[1][1] + 1):
                men_mark = [0, 1]
            # end if
            mark.append(np.array(men_mark))

        # generate overall feature ndarray
        feature_vec = []
        for i in range(len(vector_sum)):
            v = np.concatenate((vector_sum[i], pos_vecs[i], mark[i]), axis=0)
            feature_vec.append(v)
        if len(feature_vec) < 97:
            for m in range(len(feature_vec)+1,98):
                feature_vec.append([0]*len(v))
        feature_vec = np.array(feature_vec)
        print(relation_instance['content'])
        return feature_vec
    #end def
#end class
