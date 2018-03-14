__all__ = ['LexicalFeaturesTransformer']

import numpy as np

from ycml.transformers import PureTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import count_tokens
from ..utils import iter_annotations
from ..utils import find_token_span


class LexicalFeaturesTransformer(PureTransformer):
    def __init__(self, **kwargs):
        kwargs.setdefault('nparray_dtype', np.object)
        super().__init__(**kwargs)
    #end def

    def transform_one(self, relation_instance):

        m1_dict, m2_dict = relation_instance['paired_mentions']['m1'], relation_instance['paired_mentions']['m2']

        corenlp_annotations = relation_instance['corenlp_annotations']
        sentences = corenlp_annotations['sentences']

        m1_span = find_token_span(corenlp_annotations, m1_dict)
        m2_span = find_token_span(corenlp_annotations, m2_dict)

        features = {}

        if m1_span[0][1] > 0:
            features['m1_before_lemma'] = sentences[m1_span[0][0]]['tokens'][m1_span[0][1] - 1]['lemma']
        if m1_span[0][1] > 1:
            features['m1_2_before_lemma'] = sentences[m1_span[0][0]]['tokens'][m1_span[0][1] - 2]['lemma']

        if m1_span[1][1] < len(sentences[m1_span[1][0]]['tokens']) - 1:
            features['m1_after_lemma'] = sentences[m1_span[1][0]]['tokens'][m1_span[1][1] + 1]['lemma']
        if m1_span[1][1] < len(sentences[m1_span[1][0]]['tokens']) - 2:
            features['m1_2_after_lemma'] = sentences[m1_span[1][0]]['tokens'][m1_span[1][1] + 2]['lemma']

        if m2_span[0][1] > 0:
            features['m2_before_lemma'] = sentences[m2_span[0][0]]['tokens'][m2_span[0][1] - 1]['lemma']
        if m2_span[0][1] > 1:
            features['m2_2_before_lemma'] = sentences[m2_span[0][0]]['tokens'][m2_span[0][1] - 2]['lemma']

        if m2_span[1][1] < len(sentences[m2_span[1][0]]['tokens']) - 1:
            features['m2_after_lemma'] = sentences[m2_span[1][0]]['tokens'][m2_span[1][1] + 1]['lemma']
        if m2_span[1][1] < len(sentences[m2_span[1][0]]['tokens']) - 2:
            features['m2_2_after_lemma'] = sentences[m2_span[1][0]]['tokens'][m2_span[1][1] + 2]['lemma']

        features['mention_token_distance'] = count_tokens(corenlp_annotations, m1_span[0], m2_span[0], inclusive=False)
        if m1_dict['start_char'] < m2_dict['start_char']:
            features['mention_char_distance'] = m2_dict['start_char'] - m1_dict['end_char']
            features['m1_before_m2'] = True
        else:
            features['mention_char_distance'] = m1_dict['start_char'] - m2_dict['end_char']
            features['m1_before_m2'] = False
        #end if
        assert(features['mention_char_distance'] > 0)

        features['m1_token_count'] = count_tokens(corenlp_annotations, m1_span[0], m1_span[1])
        features['m2_token_count'] = count_tokens(corenlp_annotations, m2_span[0], m2_span[1])
        features['m1_char_count'] = m1_dict['end_char'] - m1_dict['start_char']
        features['m2_char_count'] = m2_dict['end_char'] - m2_dict['start_char']

        features['punctuation_marks_in_between_mentions'] = 0
        for token in iter_annotations(corenlp_annotations, m1_span[0], m2_span[0], inclusive=False):
            if token['pos'] == '.':  # might want something more robust?
                features['punctuation_marks_in_between_mentions'] += 1

        return features
    #end def
#end class
